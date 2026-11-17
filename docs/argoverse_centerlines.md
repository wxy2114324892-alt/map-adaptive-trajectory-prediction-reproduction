# Visualizing Argoverse 1 candidate centerlines

This experiment uses the official Argoverse 1 map API to retrieve candidate lane centerlines from the target vehicle's first 20 observed positions. It plots those candidates alongside the target history and the recorded future. The future is shown only for visualization; it is not passed to candidate selection. This is data exploration, not a trained trajectory prediction or a reproduction of the paper's reported metrics.

## Setup

Place the Argoverse forecasting sample and full map files under `data/raw/`, as in the default paths of `scripts/visualize_centerlines.py`. Install the project's existing Python, PyTorch, pandas and matplotlib dependencies. Clone the official API locally:

```powershell
New-Item -ItemType Directory -Force third_party
git clone --depth 1 https://github.com/argoverse/argoverse-api.git third_party/argoverse-api
```

The upstream API's installer rejects Windows. On Windows the script imports the cloned source directly. In the tested Python 3.10 environment the map imports also required SciPy, Shapely, OpenCV, descartes, imageio, OmegaConf, and colour. Install compatible versions in a dedicated virtual environment; the upstream API does not officially support Windows and this is not a guarantee of compatibility.

```powershell
.\.venv\Scripts\python.exe scripts\visualize_centerlines.py
```

Use `--scene`, `--map-dir`, or `--output` to change the paths. The default output is `outputs/argoverse_2645_centerlines.png`, which is local and ignored by Git. The default sample in this environment returned two candidates for Miami. Different maps, API versions and samples can yield different results.

Do not commit downloaded datasets, generated outputs, virtual environments or third-party source. Make sure `/third_party/` is in your local `.gitignore` before cloning.
