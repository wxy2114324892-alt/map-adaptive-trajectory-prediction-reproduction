"""Prepare matched candidate features, train a scorer, and evaluate held-out scenes."""
import argparse
import hashlib
import io
import json
import random
import sqlite3
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_constant_velocity import constant_velocity
from src.data.scenes import predict
from src.models.candidate_scorer import CandidateScorer, candidate_features
from src.models.predictor import MapAdaptiveTrajectoryPredictor
from src.utils.mode_selection import choose_history_consistent


def pack(value):
    result=io.BytesIO(); torch.save(value,result); return result.getvalue()


def unpack(blob):
    return torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)


def build_record(scene, predictions):
    """Future is used only to form supervised labels and evaluation metrics."""
    history, future, dt = scene['histories'][0], scene['future'], scene['intervals']
    if future.shape != (30, 2) or not torch.isfinite(future).all():
        raise ValueError('Ground truth must contain 30 finite XY positions')
    reference=constant_velocity(history, dt, len(future))
    features=candidate_features(history, predictions, reference)
    displacement=torch.linalg.vector_norm(predictions-future.unsqueeze(0), dim=-1)
    baseline_error=torch.linalg.vector_norm(reference-future, dim=-1)
    return {'features': features, 'ade': displacement.mean(dim=-1), 'fde': displacement[:,-1],
            'baseline': torch.stack((baseline_error.mean(), baseline_error[-1])),
            'heuristic': choose_history_consistent(predictions, reference)}


def collate(records, device):
    max_modes=max(len(r['ade']) for r in records)
    features=torch.zeros(len(records),max_modes,163,device=device)
    mask=torch.zeros(len(records),max_modes,dtype=torch.bool,device=device)
    labels=torch.empty(len(records),dtype=torch.long,device=device)
    for i,record in enumerate(records):
        count=len(record['ade']); features[i,:count]=record['features'].to(device)
        mask[i,:count]=True; labels[i]=record['ade'].argmin()
    return features,mask,labels


def batch_loss(model, records, device):
    features,mask,labels=collate(records,device)
    logits=model(features).masked_fill(~mask, -1e9)
    return F.cross_entropy(logits,labels)


def summary(rows):
    if not rows:
        raise ValueError('No validation scenes')
    result={}
    for key in ('constant_velocity','heuristic','learned','oracle'):
        ade=np.array([r[key][0] for r in rows]); fde=np.array([r[key][1] for r in rows])
        result[key]={'mean_ADE_m':float(ade.mean()),'mean_FDE_m':float(fde.mean()),
                     'median_ADE_m':float(np.median(ade)), 'median_FDE_m':float(np.median(fde)),
                     'miss_rate_over_2m':float(np.mean(fde>2))}
    return result


def make_splits(train, val, train_count, tune_count, val_count, seed):
    """Select reproducible, disjoint scorer train/tune subsets and validation IDs."""
    if min(train_count, tune_count) < 1 or val_count < 0:
        raise ValueError('Invalid scene counts')
    train=sorted(map(str, train)); val=sorted(map(str, val))
    if len(set(train)) != len(train) or len(set(val)) != len(val) or not val:
        raise ValueError('Duplicate or missing scene IDs')
    if len(train) < train_count+tune_count or (val_count and len(val)<val_count):
        raise ValueError('Not enough scenes for the requested split sizes')
    rng=random.Random(seed); rng.shuffle(train); rng.shuffle(val)
    return {'train':('train',train[:train_count]), 'tune':('train',train[train_count:train_count+tune_count]),
            'val':('val',val[:val_count] if val_count else val)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/official_full/checkpoint.pt')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/learned_selector')
    p.add_argument('--train-count', type=int, default=10000)
    p.add_argument('--tune-count', type=int, default=2000)
    p.add_argument('--val-count', type=int, default=0, help='0 evaluates the entire official validation split')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    args=p.parse_args()
    if min(args.train_count,args.tune_count,args.epochs,args.batch_size)<1 or args.val_count<0: p.error('Counts must be positive (val-count may be zero)')
    args.output.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    config={'version':1,'checkpoint_sha256':sha,'train_count':args.train_count,'tune_count':args.tune_count,
            'val_count':args.val_count,'seed':args.seed}
    db=sqlite3.connect(args.output/'features.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS records (part TEXT NOT NULL,id TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(part,id))')
    old=db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if old and json.loads(old[0]) != config: raise ValueError('Feature cache configuration differs; choose another output directory')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('config',?)",(json.dumps(config),)); db.commit()
    checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    predictor=MapAdaptiveTrajectoryPredictor(agent_input_dim=4).to(args.device)
    predictor.load_state_dict(checkpoint['model']); predictor.eval()
    source=sqlite3.connect(f'file:{args.cache.resolve().as_posix()}?mode=ro',uri=True)
    train=sorted(str(x[0]) for x in source.execute("SELECT id FROM scenes WHERE split='train'"))
    val=sorted(str(x[0]) for x in source.execute("SELECT id FROM scenes WHERE split='val'"))
    if set(train)!=set(map(str,checkpoint['train_ids'])) or set(val)!=set(map(str,checkpoint['val_ids'])):
        raise ValueError('Scene IDs do not match predictor checkpoint splits')
    parts=make_splits(train,val,args.train_count,args.tune_count,args.val_count,args.seed)
    for part,(split,ids) in parts.items():
        with torch.no_grad():
            for n,sid in enumerate(ids,1):
                if not db.execute('SELECT 1 FROM records WHERE part=? AND id=?',(part,sid)).fetchone():
                    blob=source.execute('SELECT payload FROM scenes WHERE split=? AND id=?',(split,sid)).fetchone()[0]
                    scene=unpack(blob); predictions=predict(predictor,scene,args.device).cpu()
                    db.execute('INSERT INTO records VALUES (?,?,?)',(part,sid,pack(build_record(scene,predictions))))
                if n%500==0:
                    db.commit(); print(f'Preparing {part}: {n}/{len(ids)}',flush=True)
            db.commit(); print(f'Prepared {part}: {len(ids)}',flush=True)
    del predictor
    torch.manual_seed(args.seed)
    scorer=CandidateScorer().to(args.device)
    optimizer=torch.optim.Adam(scorer.parameters(),lr=.001)
    train_records=[unpack(db.execute('SELECT payload FROM records WHERE part=? AND id=?',('train',sid)).fetchone()[0]) for sid in parts['train'][1]]
    tune_records=[unpack(db.execute('SELECT payload FROM records WHERE part=? AND id=?',('tune',sid)).fetchone()[0]) for sid in parts['tune'][1]]
    best=float('inf'); best_epoch=0; history=[]; best_state=None
    for epoch in range(args.epochs):
        scorer.train(); order=list(range(len(train_records))); random.Random(args.seed+epoch).shuffle(order); loss_sum=0.
        for start in range(0,len(order),args.batch_size):
            batch=[train_records[i] for i in order[start:start+args.batch_size]]
            optimizer.zero_grad(); loss=batch_loss(scorer,batch,args.device); loss.backward(); optimizer.step(); loss_sum+=loss.item()*len(batch)
        scorer.eval(); tuning=0.
        with torch.no_grad():
            for start in range(0,len(tune_records),args.batch_size):
                batch=tune_records[start:start+args.batch_size]; tuning+=batch_loss(scorer,batch,args.device).item()*len(batch)
        train_loss=loss_sum/len(train_records); tune_loss=tuning/len(tune_records)
        history.append({'epoch':epoch+1,'train_cross_entropy':train_loss,'tune_cross_entropy':tune_loss})
        if tune_loss<best:
            best=tune_loss; best_epoch=epoch+1; best_state={k:v.detach().cpu().clone() for k,v in scorer.state_dict().items()}
        print(f'Epoch {epoch+1}: train={train_loss:.4f}, tune={tune_loss:.4f}',flush=True)
    scorer.load_state_dict(best_state); scorer.eval()
    torch.save({'model':best_state,'config':config,'best_epoch':best_epoch,'history':history,'feature_size':163},args.output/'scorer.pt')
    restored=torch.load(args.output/'scorer.pt', map_location=args.device, weights_only=True)
    scorer.load_state_dict(restored['model']); scorer.eval()
    rows=[]
    with torch.no_grad():
        for sid in parts['val'][1]:
            record=unpack(db.execute('SELECT payload FROM records WHERE part=? AND id=?',('val',sid)).fetchone()[0])
            selected=int(scorer(record['features'].to(args.device)).argmax().item())
            ade,fde=record['ade'],record['fde']; h=record['heuristic']
            rows.append({'id':sid,'modes':len(ade),'selected_mode':selected,
                         'constant_velocity':record['baseline'].tolist(),'heuristic':[ade[h].item(),fde[h].item()],
                         'learned':[ade[selected].item(),fde[selected].item()], 'oracle':[ade.min().item(),fde.min().item()]})
    result={'config':config,'validation_count':len(rows),'best_epoch':best_epoch,'training_history':history,
            'methods':summary(rows),'scenes':rows}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scenes','training_history')},indent=2))
    source.close(); db.close()


if __name__=='__main__': main()
