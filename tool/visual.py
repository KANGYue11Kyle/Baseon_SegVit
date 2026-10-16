"""
可视化SegViT预测结果 - 输出元素多且准确率高的图片
- 输入：iter_45000.pth checkpoint
- 输出：前景占比高且准确率高的图片的可视化结果
"""

import os
import sys
import math

# 添加项目根目录到路径
sys.path.insert(0, r'F:\download\SegVit-master')

# ========== 关键：先导入自定义模块，注册模型 ==========
from decode_heads import atm_head, tpn_atm_head
from losses import atm_loss
from backbone import vit_shrink
# ====================================================

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mmseg.apis import inference_segmentor, init_segmentor
from tqdm import tqdm

# ========== Monkey Patch: 临时修复 d3_to_d4 ==========
def fixed_d3_to_d4(self, t):
    """动态计算特征图尺寸，避免硬编码"""
    n, hw, c = t.size()
    h = int(math.sqrt(hw))
    while hw % h != 0:
        h -= 1
    w = hw // h
    return t.transpose(1, 2).reshape(n, c, h, w)

# 替换 ATMHead 中的 d3_to_d4 方法
atm_head.ATMHead.d3_to_d4 = fixed_d3_to_d4
print("已临时修复 d3_to_d4 函数")
# ====================================================

# ========== 配置 ==========
config_file = r'F:\download\SegVit-master\configs\segvit\segvit_vit-base_surgery_continue_4_23.py'
checkpoint_file = r'F:\download\SegVit-master\work_dirs\segvit_vit-base_surgery_continue_4_24_night\iter_45000.pth'
output_dir = r'F:\download\SegVit-master\visual_best'
os.makedirs(output_dir, exist_ok=True)

# ========== 参数设置 ==========
NUM_BEST_IMAGES = 10   # 输出图片数量
NUM_SHOW = 15          # 综合排序时展示的图片数量
USE_COLORFUL = True    # 使用彩色调色板

# 综合评分权重
WEIGHT_ACC = 0.6       # 准确率权重（越高越重要）
WEIGHT_FOREGROUND = 0.4  # 前景占比权重（越高越重要）

# ========== 彩色调色板 ==========
palette_colorful = [
    [0, 0, 0],           # 0: background - 黑色
    [255, 0, 0],         # 1: tool1_manipulator - 红色
    [200, 100, 0],       # 2: tool1_shaft - 橙色
    [0, 255, 0],         # 3: tool2_manipulator - 绿色
    [0, 200, 100],       # 4: tool2_shaft - 青绿
    [0, 0, 255],         # 5: tool3_manipulator - 蓝色
    [100, 0, 200],       # 6: tool3_shaft - 紫色
    [255, 255, 0],       # 7: tool4_manipulator - 黄色
    [255, 0, 255],       # 8: tool4_shaft - 洋红
    [0, 255, 255],       # 9: uterus - 青色
]

palette_grayscale = [
    [0, 0, 0], [20, 20, 20], [40, 40, 40], [60, 60, 60], [80, 80, 80],
    [100, 100, 100], [120, 120, 120], [140, 140, 140], [160, 160, 160], [180, 180, 180]
]

palette_rgb = palette_colorful if USE_COLORFUL else palette_grayscale
print(f"使用{'彩色' if USE_COLORFUL else '灰度'}调色板")

class_names = [
    'background', 'tool1_manipulator', 'tool1_shaft',
    'tool2_manipulator', 'tool2_shaft', 'tool3_manipulator',
    'tool3_shaft', 'tool4_manipulator', 'tool4_shaft', 'uterus'
]

# ========== 获取测试集图片 ==========
test_img_dir = 'data/surgery/images/testing'
print(f"检查测试集路径: {test_img_dir}")

if os.path.exists(test_img_dir):
    test_images = list(Path(test_img_dir).glob('*.jpg')) + list(Path(test_img_dir).glob('*.JPG'))
else:
    alt_paths = [
        'F:/download/SegVit-master/data/surgery/images/validation',
        'data/surgery/images/validation',
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            test_img_dir = alt
            test_images = list(Path(alt).glob('*.jpg')) + list(Path(alt).glob('*.JPG'))
            break
    else:
        test_images = []

print(f"找到 {len(test_images)} 张图片")

if len(test_images) == 0:
    print("错误: 没有找到任何测试图片！")
    sys.exit(1)

# ========== 加载模型 ==========
print("\n加载模型中...")
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
print("模型加载完成！")

# ========== 辅助函数 ==========
def compute_pixel_accuracy(pred_mask, gt_mask):
    """计算像素准确率"""
    if gt_mask is None:
        return 0.0
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    correct = np.sum(pred_mask == gt_mask)
    total = pred_mask.size
    return correct / total if total > 0 else 0.0

def compute_foreground_ratio(mask):
    """计算前景占比（非背景像素的比例）"""
    foreground = np.sum(mask > 0)  # 非0像素
    total = mask.size
    return foreground / total if total > 0 else 0.0

def compute_composite_score(acc, fg_ratio, w_acc=WEIGHT_ACC, w_fg=WEIGHT_FOREGROUND):
    """计算综合评分（准确率 + 前景占比）"""
    return w_acc * acc + w_fg * fg_ratio

def load_gt_mask(img_path):
    """加载对应的真实掩码"""
    mask_path = Path(str(img_path).replace('images', 'annotations').replace('.jpg', '.png'))
    if not mask_path.exists():
        mask_path = Path('data/surgery/annotations/testing') / f"{img_path.stem}.png"
        if not mask_path.exists():
            return None
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return gt_mask

# ========== 计算每张图片的指标 ==========
print("\n计算每张图片的指标...")
image_scores = []

for img_path in tqdm(test_images, desc="评估中"):
    try:
        # 加载真实掩码
        gt_mask = load_gt_mask(img_path)
        if gt_mask is None:
            continue

        # 推理
        result = inference_segmentor(model, str(img_path))
        pred_mask = result[0]

        # 计算指标
        acc = compute_pixel_accuracy(pred_mask, gt_mask)
        fg_ratio = compute_foreground_ratio(gt_mask)
        composite_score = compute_composite_score(acc, fg_ratio)

        image_scores.append({
            'img_path': img_path,
            'pred_mask': pred_mask,
            'gt_mask': gt_mask,
            'acc': acc,
            'fg_ratio': fg_ratio,
            'composite_score': composite_score
        })
    except Exception as e:
        print(f"处理 {img_path.name} 时出错: {e}")

# ========== 按综合评分排序 ==========
image_scores.sort(key=lambda x: x['composite_score'], reverse=True)
best_images = image_scores[:NUM_BEST_IMAGES]

print(f"\n综合评分最高的 {NUM_BEST_IMAGES} 张图片 (准确率权重={WEIGHT_ACC}, 前景占比权重={WEIGHT_FOREGROUND}):")
print(f"{'排名':<4} {'文件名':<30} {'准确率':<10} {'前景占比':<10} {'综合评分':<10}")
print("-" * 65)
for i, img in enumerate(best_images):
    print(f"{i+1:<4} {img['img_path'].name:<30} {img['acc']*100:<10.2f}% {img['fg_ratio']*100:<10.2f}% {img['composite_score']*100:<10.2f}")

# ========== 可视化函数 ==========
def visualize_prediction(img_info, save_path):
    """可视化预测结果"""
    img_path = img_info['img_path']
    pred_mask = img_info['pred_mask']
    gt_mask = img_info['gt_mask']
    acc = img_info['acc']
    fg_ratio = img_info['fg_ratio']

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建彩色掩码
    colored_pred = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette_rgb):
        colored_pred[pred_mask == class_id] = color

    colored_gt = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette_rgb):
        colored_gt[gt_mask == class_id] = color

    overlay = cv2.addWeighted(img, 0.6, colored_pred, 0.4, 0)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(colored_gt)
    axes[0, 1].set_title(f'Ground Truth (前景占比: {fg_ratio*100:.1f}%)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(colored_pred)
    axes[1, 0].set_title(f'SegViT Prediction (准确率: {acc*100:.1f}%)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')

    plt.suptitle(f'{img_path.name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ========== 生成可视化结果 ==========
print("\n生成可视化结果...")
for i, img_info in enumerate(best_images):
    acc = img_info['acc']
    fg = img_info['fg_ratio']
    save_path = os.path.join(output_dir, f'{i+1:02d}_{img_info["img_path"].stem}_acc_{acc*100:.1f}_fg_{fg*100:.1f}.jpg')
    visualize_prediction(img_info, save_path)
    print(f"  已保存: {os.path.basename(save_path)}")

# ========== 生成汇总图 ==========
print("\n生成汇总图...")
n_cols = min(5, NUM_BEST_IMAGES)
n_rows = (NUM_BEST_IMAGES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 5, n_rows * 4))

for idx, img_info in enumerate(best_images[:n_rows * n_cols]):
    row = idx // n_cols
    col = (idx % n_cols) * 2

    # 原图
    img = cv2.imread(str(img_info['img_path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[row, col].imshow(img)
    axes[row, col].set_title(f"{img_info['img_path'].stem}\nAcc:{img_info['acc']*100:.1f}% FG:{img_info['fg_ratio']*100:.1f}%", fontsize=8)
    axes[row, col].axis('off')

    # 预测掩码
    colored_pred = np.zeros((*img_info['pred_mask'].shape, 3), dtype=np.uint8)
    for cid, color in enumerate(palette_rgb):
        colored_pred[img_info['pred_mask'] == cid] = color
    axes[row, col + 1].imshow(colored_pred)
    axes[row, col + 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'best_summary_grid.jpg'), dpi=150, bbox_inches='tight')
plt.close()
print(f"汇总图保存至: {output_dir}/best_summary_grid.jpg")

# ========== 生成统计报告 ==========
print("\n生成统计报告...")
avg_acc = sum([s['acc'] for s in image_scores]) / len(image_scores) if image_scores else 0
avg_fg = sum([s['fg_ratio'] for s in image_scores]) / len(image_scores) if image_scores else 0

with open(os.path.join(output_dir, 'statistics.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("SegViT 评估报告 - 综合评分排名\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"权重设置:\n")
    f.write(f"  准确率权重: {WEIGHT_ACC}\n")
    f.write(f"  前景占比权重: {WEIGHT_FOREGROUND}\n\n")
    f.write(f"整体统计:\n")
    f.write(f"  评估图片数: {len(image_scores)}\n")
    f.write(f"  平均准确率: {avg_acc*100:.2f}%\n")
    f.write(f"  平均前景占比: {avg_fg*100:.2f}%\n\n")
    f.write("=" * 70 + "\n")
    f.write("Top 15 综合评分排名\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'排名':<4} {'文件名':<35} {'准确率':<12} {'前景占比':<12} {'综合评分':<12}\n")
    f.write("-" * 75 + "\n")
    for i, img in enumerate(image_scores[:NUM_SHOW]):
        f.write(f"{i+1:<4} {img['img_path'].name:<35} {img['acc']*100:<12.2f}% {img['fg_ratio']*100:<12.2f}% {img['composite_score']*100:<12.2f}\n")

print(f"统计报告保存至: {output_dir}/statistics.txt")

# 打印摘要
print(f"\n📊 统计摘要:")
print(f"  总图片数: {len(image_scores)}")
print(f"  平均准确率: {avg_acc*100:.2f}%")
print(f"  平均前景占比: {avg_fg*100:.2f}%")
print(f"  最高准确率: {image_scores[0]['acc']*100:.2f}%")
print(f"  最高前景占比: {max(image_scores, key=lambda x: x['fg_ratio'])['fg_ratio']*100:.2f}%")

print(f"\n✅ 完成！结果保存至: {output_dir}")