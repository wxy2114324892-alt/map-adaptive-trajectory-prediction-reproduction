\# 地图自适应多模态轨迹预测复现



本项目是对以下论文的非官方 PyTorch 复现：



> Map-Adaptive Multimodal Trajectory Prediction Using Hierarchical Graph Neural Networks  

> IEEE Robotics and Automation Letters, 2023



\## 项目状态



正在开发中。



当前环境：



\- PyTorch 2.7.1

\- CUDA 12.8

\- PyTorch Geometric 2.6.1

\- NVIDIA GeForce RTX 5060 Laptop GPU



\## 复现目标



\- 使用 GRU 编码车辆历史运动轨迹；

\- 编码候选车道中心线（Candidate Centerlines，CCLs）；

\- 实现分层图算子（Hierarchical Graph Operator，HGO）；

\- 生成以下三类预测轨迹：

&#x20; - CCL-following：沿候选中心线行驶；

&#x20; - Scene-reasoning：根据完整场景进行推理；

&#x20; - Motion-maintaining：保持当前运动趋势；

\- 使用 minADE、minFDE 等指标评估预测结果。



\## 项目结构



\- `src/data/`：数据读取与预处理

\- `src/models/`：模型组件与 HGO

\- `src/utils/`：评估指标和可视化

\- `configs/`：实验配置

\- `scripts/`：训练与评估脚本

\- `docs/`：论文笔记和实现说明

\- `tests/`：单元测试



\## 说明



本项目仅用于学习、研究和个人作品展示，并非论文作者提供的官方实现。项目将尽量依据论文描述复现模型结构与实验流程，并明确记录无法完全确认的实现细节。

