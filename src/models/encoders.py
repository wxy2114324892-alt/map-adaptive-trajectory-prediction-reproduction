"""使用 GRU 编码车辆历史轨迹和候选车道中心线。"""

# 导入 PyTorch。
# torch 用于创建和处理张量，例如一批车辆的历史坐标。
import torch

# nn 中包含构建神经网络需要的模块，例如 GRU、Linear 等。
from torch import nn


class GRUSequenceEncoder(nn.Module):
    """
    使用 GRU 将一段有顺序的输入序列编码成一个固定长度的特征向量。

    这个类之后可以用于编码两种序列:

    1. 车辆历史轨迹:
       [(x1, y1), (x2, y2), ..., (xT, yT)]

    2. 候选车道中心线 CCL:
       [(x1, y1), (x2, y2), ..., (xL, yL)]
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
        初始化 GRU 编码器。

        Args:
            input_dim:
                每个时间点包含的输入特征数量。

                例如，每个历史点只有二维坐标 (x, y) 时：
                input_dim = 2

                如果以后加入速度 vx、vy,则输入可以是:
                (x, y, vx, vy)
                此时 input_dim = 4。

            hidden_dim:
                GRU 隐藏状态的维度，也就是最终输出特征的维度。

                例如 hidden_dim = 64，表示 GRU 会把整段序列
                总结成一个 64 维特征向量。
        """

        # 调用父类 nn.Module 的初始化方法。
        # 所有自定义 PyTorch 模型都需要执行这一句。
        super().__init__()

        # 创建 GRU 网络。
        self.gru = nn.GRU(
            # 每个时间点输入多少个特征。
            input_size=input_dim,

            # GRU 隐藏状态的维度。
            # 最终会用这个隐藏状态表示整段输入序列。
            hidden_size=hidden_dim,

            # 输入张量的第一维表示 batch。
            #
            # 因此输入形状为：
            # [batch_size, sequence_length, input_dim]
            #
            # 例如：
            # [8, 20, 2]
            #
            # 表示一批有 8 辆车，
            # 每辆车包含 20 个历史位置，
            # 每个位置有 x、y 两个特征。
            batch_first=True,
        )
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        定义输入序列经过 GRU 后的计算过程。

        Args:
            sequence:
                输入序列，形状为：

                [batch_size, sequence_length, input_dim]

                例如车辆历史轨迹的形状为 [8, 20, 2]：
                - 8：一批有 8 辆车；
                - 20：每辆车有 20 个历史时间点；
                - 2：每个点包含 x、y 两个特征。

        Returns:
            每条序列对应的最终特征，形状为：

            [batch_size, hidden_dim]

            如果 batch_size = 8、hidden_dim = 64，
            那么输出形状就是 [8, 64]。
        """

        # 将完整序列输入 GRU。
        #
        # GRU 会返回两个结果：
        #
        # 1. all_outputs：
        #    每个时间步对应的隐藏状态。
        #    形状为：
        #    [batch_size, sequence_length, hidden_dim]
        #
        # 2. final_hidden：
        #    GRU 阅读完整段序列后得到的最终隐藏状态。
        #    形状为：
        #    [num_layers, batch_size, hidden_dim]
        all_outputs, final_hidden = self.gru(sequence)

        # 当前只使用一层 GRU。
        #
        # final_hidden[-1] 表示取最后一层 GRU 的最终隐藏状态，
        # 其形状从：
        # [num_layers, batch_size, hidden_dim]
        #
        # 变为：
        # [batch_size, hidden_dim]
        #
        # 这个向量就是对整段历史轨迹或整条 CCL 的总结。
        encoded_feature = final_hidden[-1]

        return encoded_feature

class AgentHistoryEncoder(GRUSequenceEncoder):
    """
    车辆历史轨迹编码器。

    输入是一辆车过去若干时刻的运动状态，例如：

    [(x1, y1), (x2, y2), ..., (xT, yT)]

    输出是一个固定长度的车辆运动特征，用来描述：
    - 车辆当前朝哪个方向运动；
    - 车辆大致速度如何；
    - 车辆是在直行、转弯还是减速；
    - 过去的运动趋势。
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        """
        Args:
            input_dim:
                每个历史时刻的车辆状态维度。
                当前默认只使用二维坐标 (x, y)，所以默认值为 2。

            hidden_dim:
                输出的车辆运动特征维度，默认是 64。
        """

        # 调用父类 GRUSequenceEncoder，
        # 创建属于车辆历史轨迹编码器自己的 GRU。
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )


class CenterlineEncoder(GRUSequenceEncoder):
    """
    候选车道中心线编码器。

    输入是一条候选中心线上的有序坐标点，例如：

    [(x1, y1), (x2, y2), ..., (xL, yL)]

    输出是一个固定长度的车道特征，用来描述：
    - 车道整体延伸方向；
    - 车道是直线还是曲线；
    - 车道弯曲趋势；
    - 该 CCL 所代表的潜在行驶方向。
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        """
        Args:
            input_dim:
                CCL 上每个点的特征维度。
                当前每个点只有二维坐标 (x, y)，默认值为 2。

            hidden_dim:
                输出的车道特征维度，默认是 64。
        """

        # 调用父类 GRUSequenceEncoder，
        # 创建属于 CCL 编码器自己的 GRU。
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )