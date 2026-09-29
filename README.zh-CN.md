# LuxNet ML RDC

<p align="center">
  <b>多模态照度场预测与自然光响应式实时调光</b>
</p>

![LuxNet 照明控制框架](docs/assets/overview_workflow.png)

LuxNet 是一个面向自然光与人工照明协同控制的深度学习项目。项目使用多模态代理模型预测各灯具产生的照度分布，再通过粒子群算法计算适合当前自然光状态的调光比例。

本仓库提供从数据准备、模型训练、验证测试到照明优化案例的精简流程。

## 论文信息

本仓库对应以下论文：

> Qiang Zeng, Tao Wang, Yingying Zhang, Shaojun Zhu, Junhao Xu, and Yushuai Zhao. **Real-time dimming control of integrated daylight and electric lighting systems using a multimodal generative surrogate model.** *Building and Environment*, 305, 115203, 2026. [DOI: 10.1016/j.buildenv.2026.115203](https://doi.org/10.1016/j.buildenv.2026.115203) · Web of Science 入藏号：`WOS:001874521300001`

## 方法简介

代理模型采用多模态残差 U-Net。模型输入包括灯具位置、房间区域和窗户投影三个空间通道，以及描述房间、窗户、灯具和表面属性的 11 维物理条件向量。

模型输出 128 × 128 的人工照明照度图。多灯房间先逐灯预测，再通过线性叠加得到整体人工照明分布。粒子群算法根据自然光分布和目标照度搜索灯具调光比例。

![MResUNet 网络结构](docs/assets/model_architecture.png)

## 论文结果展示

下图直接采用论文 Fig. 7(a)，展示一天中自然光变化时，优化后的总照度分布和灯具调光比例如何随之调整。

![论文 Fig. 7(a) 代表性调光结果](docs/assets/paper_fig7a_case15.png)

下图直接采用论文 Fig. 15。房间不同区域设置不同目标照度，控制器据此调整各灯具输出。

![论文 Fig. 15 个性化目标调光结果](docs/assets/paper_fig15_personalized.png)

## 快速运行

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -v
python evaluate.py --split test
python demo.py --case case01
python demo.py --case case07 --personalized
```

运行结果保存在 `outputs/`。

## 完整数据和训练

从 GitHub Release 下载 `luxnet_dataset_4000.zip`，然后运行：

```bash
python scripts/extract_dataset.py path/to/luxnet_dataset_4000.zip
python evaluate.py --data data/full --splits data/splits.json --split test
python train.py --data data/full --splits data/splits.json --output outputs/train_01
```

数据文件包括模型图像输入、物理条件向量、照度标签和场景信息，具体定义见 [DATASET.md](DATASET.md)。

## 训练过程

仓库保留了训练过程曲线，并提供可以直接进行测试和案例演示的预训练权重。

![训练曲线](docs/assets/training_curve.png)

## 文件夹说明

```text
luxnet/       模型、数据编码、损失函数、指标和优化代码
data/         数据划分文件和可直接运行的小样本
examples/     两个照明调光案例
results/      训练过程和已有测试结果
scripts/      数据集工具
tests/        模型与优化算法的基础检查
weights/      预训练模型权重
train.py      模型训练入口
evaluate.py   验证与测试入口
demo.py       完整调光案例入口
```

完整引用信息见 `CITATION.cff`。

