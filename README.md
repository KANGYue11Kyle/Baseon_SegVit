# Baseon_SegVit
# 基于 SegViT 的手术器械腔体分割

基于 [SegViT](https://github.com/zbwxp/SegViT)（Segmentation with Vision Transformer）实现的手术器械腔体分割项目。采用轻量化 ViT 骨干网络，结合分割解码头与自定义损失函数，支持多卡训练、数据集预处理、训练/测试与可视化全流程。

## 📁 仓库文件说明

```
.
├── backbone/                                       # 轻量化 ViT 骨干网络定义代码，负责图像特征提取
├── decode_heads/                                   # 分割解码头模块
├── losses/                                         # 自定义损失函数模块
├── tools/                                          # 全流程工具脚本
│   ├── dist*.sh                                    # 多卡训练启动脚本
│   ├── pre_data_train_val.py                       # 数据集预处理
│   ├── train.py                                    # 模型训练
│   ├── test.py                                     # 模型测试
│   └── visual.py                                   # 生成分割可视化结果
├── best.pth                                        # 基于验证集指标自动保存的最优二进制权重文件
├── segvit.txt                                      # 本次训练全流程实验日志与参数记录
├── segvit_4_26.py                                  # 模型核心配置文件
├── 05_215125_acc_90.0_fg_35.7.jpg                  # 最优模型精度指标截图
└── attention_tool2_manipulator.png                 # 注意力机制可视化结果图
```

## 🔍 模块说明

| 文件 / 目录 | 说明 |
| --- | --- |
| `backbone/` | 存放轻量化 ViT 骨干网络定义代码，负责图像特征提取 |
| `decode_heads/` | 分割解码头模块 |
| `losses/` | 自定义损失函数模块 |
| `tools/` | 全流程工具脚本 |
| `tools/dist*.sh` | 多卡训练启动脚本 |
| `tools/pre_data_train_val.py` | 数据集预处理 |
| `tools/train.py` | 模型训练 |
| `tools/test.py` | 模型测试 |
| `tools/visual.py` | 生成分割可视化结果 |
| `best.pth` | 训练过程中基于验证集指标自动保存的最优二进制权重文件 |
| `segvit.txt` | 本次训练全流程实验日志与参数记录 |
| `segvit_4_26.py` | 模型核心配置文件，定义网络结构、输入尺寸、类别数等超参数 |
| `05_215125_acc_90.0_fg_35.7.jpg` | 最优模型精度指标截图 |
| `attention_tool2_manipulator.png` | 注意力机制可视化结果图 |

## 🧠 模型结构

- **主干网络（Backbone）**：轻量化 ViT（Vision Transformer），负责图像特征提取
- **解码头（Decode Heads）**：分割解码头模块
- **损失函数（Losses）**：自定义损失函数
- **核心配置**：`segvit_4_26.py`，定义网络结构、输入尺寸、类别数等超参数

## 🚀 快速开始

### 环境依赖

> TODO：补充 Python、PyTorch、CUDA 版本及依赖安装命令

```bash
# 示例
pip install -r requirements.txt
```

### 数据集预处理

```bash
python tools/pre_data_train_val.py
```

### 模型训练

**单卡训练：**

```bash
python tools/train.py --cfg segvit_4_26.py
```

**多卡训练：**

```bash
bash tools/dist_train.sh
```

> TODO：补充实际脚本名与参数说明

训练过程中会基于验证集指标自动保存最优权重至 `best.pth`，全流程日志记录于 `segvit.txt`。

### 模型测试

```bash
python tools/test.py --cfg segvit_4_26.py --weight best.pth
```

### 结果可视化

```bash
python tools/visual.py --cfg segvit_4_26.py --weight best.pth
```

## 📊 训练结果

最优模型精度指标：

![最优模型精度](05_215125_acc_90.0_fg_35.7.jpg)

- **Accuracy**：90.0
- **FG（前景）指标**：35.7

> 指标截图文件名：`05_215125_acc_90.0_fg_35.7.jpg`

## 🎯 注意力可视化

模型注意力机制可视化结果（手术器械）：

![注意力可视化](attention_tool2_manipulator.png)

## 🏆 最优权重

训练过程中基于验证集指标自动保存的最优 checkpoint 为 **`best.pth`**。

## 🙏 致谢

本项目基于 [SegViT](https://github.com/zbwxp/SegViT) 实现，感谢原作者的开源工作。
