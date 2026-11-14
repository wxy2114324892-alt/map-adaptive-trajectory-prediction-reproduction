"""测试车辆历史轨迹编码器和候选中心线编码器。"""

import torch

# 从刚才编写的 encoders.py 中导入两个编码器。
from src.models.encoders import AgentHistoryEncoder, CenterlineEncoder


def main() -> None:
    """使用模拟数据测试两个 GRU 编码器。"""

    # 如果 CUDA 可用，就在 NVIDIA GPU 上运行；
    # 否则自动退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"运行设备: {device}")

    # ---------------------------------------------------------
    # 1. 创建车辆历史轨迹模拟数据
    # ---------------------------------------------------------

    # 假设一批有 8 辆车。
    agent_batch_size = 8

    # 每辆车包含过去 20 个时间点。
    history_length = 20

    # 每个时间点暂时只有 x、y 两个特征。
    agent_input_dim = 2

    # 创建形状为 [8, 20, 2] 的随机张量，
    # 并将其放到 GPU 或 CPU。
    agent_history = torch.randn(
        agent_batch_size,
        history_length,
        agent_input_dim,
        device=device,
    )

    # ---------------------------------------------------------
    # 2. 创建候选中心线模拟数据
    # ---------------------------------------------------------

    # 假设当前一共有 12 条候选中心线。
    centerline_batch_size = 12

    # 每条中心线由 30 个有序坐标点组成。
    centerline_length = 30

    # 每个中心线点包含 x、y 两个坐标。
    centerline_input_dim = 2

    # 创建形状为 [12, 30, 2] 的随机张量。
    centerlines = torch.randn(
        centerline_batch_size,
        centerline_length,
        centerline_input_dim,
        device=device,
    )

    # ---------------------------------------------------------
    # 3. 创建两个相互独立的 GRU 编码器
    # ---------------------------------------------------------

    agent_encoder = AgentHistoryEncoder(
        input_dim=agent_input_dim,
        hidden_dim=64,
    ).to(device)

    centerline_encoder = CenterlineEncoder(
        input_dim=centerline_input_dim,
        hidden_dim=64,
    ).to(device)

    # ---------------------------------------------------------
    # 4. 对两种输入序列进行编码
    # ---------------------------------------------------------

    # [8, 20, 2] -> [8, 64]
    agent_features = agent_encoder(agent_history)

    # [12, 30, 2] -> [12, 64]
    centerline_features = centerline_encoder(centerlines)

    print(f"车辆历史输入形状: {agent_history.shape}")
    print(f"车辆特征输出形状: {agent_features.shape}")

    print(f"CCL 输入形状: {centerlines.shape}")
    print(f"CCL 特征输出形状: {centerline_features.shape}")

    # ---------------------------------------------------------
    # 5. 自动检查输出形状是否正确
    # ---------------------------------------------------------

    assert agent_features.shape == (8, 64)
    assert centerline_features.shape == (12, 64)

    print("两个编码器的测试均已通过。")


# 只有直接运行该文件时，才执行 main()。
if __name__ == "__main__":
    main()