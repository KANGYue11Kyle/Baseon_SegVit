import sys
sys.path.insert(0, '.')

import torch
from mmseg.apis import init_segmentor
from mmseg.core.evaluation import eval_metrics
from mmseg.datasets import build_dataloader, build_dataset
from mmcv import Config

# 加载配置
cfg = Config.fromfile(r'F:\download\SegVit-master\configs\segvit\segvit_vit-base_surgery.py')

# 构建数据集（不带 workers_per_gpu）
dataset = build_dataset(cfg.data.test)

# 构建数据加载器（workers_per_gpu 在这里设置）
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    shuffle=False,
    dist=False
)

# 加载模型
model = init_segmentor(cfg, 'work_dirs/segvit_vit-base_surgery/latest.pth', device='cuda:0')
model.eval()

# 评估
outputs = []
print(f"开始评估 {len(data_loader)} 个批次...")
for i, data in enumerate(data_loader):
    if i % 50 == 0:
        print(f"处理中: {i}/{len(data_loader)}")
    with torch.no_grad():
        # 将数据移动到 GPU
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        result = model(return_loss=False, **data)
        outputs.append(result)

print("评估完成，计算指标...")

# 计算指标
eval_results = eval_metrics(
    dataset,
    outputs,
    ['mIoU', 'mAcc', 'aAcc'],
    logger=None
)

print("\n" + "="*50)
print("评估结果：")
print("="*50)
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
print("="*50)