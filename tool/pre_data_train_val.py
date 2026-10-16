"""
数据集划分脚本 - 随机划分（训练集260视频，测试集50视频，允许重叠）
- 训练集: 随机选择260个视频的所有帧
- 测试集: 训练集中随机选10个视频 + 剩余未选中的40个视频
- 每个视频有6帧（t=0,1,2,3,4,5）
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ==================== 请修改这三个路径 ====================
IMAGES_PATH = r"F:\download\HRNet-Semantic-Segmentation-HRNet-OCR\data\surgery\images"  # 原始图片文件夹路径
MASKS_PATH = r"F:\download\task3\masks"  # 原始掩码文件夹路径
OUTPUT_PATH = r"F:\download\SegVit-master\data"  # 输出路径
# ========================================================

# ==================== 参数设置 ====================
TOTAL_CLIPS = 300  # 总共300个视频
TRAIN_CLIPS_COUNT = 260  # 训练集视频数
TEST_CLIPS_COUNT = 50  # 测试集视频数
OVERLAP_CLIPS = 10  # 训练集和测试集重叠的视频数
RANDOM_SEED = 42  # 随机种子


# ====================================================

def set_seed(seed):
    random.seed(seed)


def get_clip_num(filename: str) -> int:
    """从文件名中提取clip编号，格式如 '001025.jpg' -> 1"""
    return int(filename[:3])


def get_frame_id(filename: str) -> int:
    """从文件名中提取帧编号，格式如 '001025.jpg' -> 25"""
    return int(filename[3:6])


def group_by_clip(file_list):
    """将文件按clip编号分组"""
    clip_dict = defaultdict(list)
    for f in file_list:
        clip_num = get_clip_num(f.name)
        clip_dict[clip_num].append(f)
    return clip_dict


def copy_files_with_progress(file_pairs, output_root, split_name, desc):
    """复制图片和对应的掩码"""
    img_dir = output_root / 'images' / split_name
    ann_dir = output_root / 'annotations' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in tqdm(file_pairs, desc=desc):
        shutil.copy2(img_path, img_dir / img_path.name)
        if mask_path:
            shutil.copy2(mask_path, ann_dir / mask_path.name)


def main():
    images_dir = Path(IMAGES_PATH)
    masks_dir = Path(MASKS_PATH)
    output_root = Path(OUTPUT_PATH)

    set_seed(RANDOM_SEED)

    # 检查路径
    if not images_dir.exists():
        raise FileNotFoundError(f"图片路径不存在: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"掩码路径不存在: {masks_dir}")

    # 获取所有文件
    all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    all_masks = {p.stem: p for p in masks_dir.glob('*.png')}

    print(f"找到 {len(all_images)} 张图片，{len(all_masks)} 个掩码")

    # 按clip分组
    clips_dict = group_by_clip(all_images)
    all_clip_nums = sorted(clips_dict.keys())
    print(f"共有 {len(all_clip_nums)} 个视频 (1-300)")

    # 随机选择训练集视频
    random.shuffle(all_clip_nums)
    train_clips = set(all_clip_nums[:TRAIN_CLIPS_COUNT])

    # 从训练集中随机选 OVERLAP_CLIPS 个作为重叠部分
    train_clips_list = list(train_clips)
    random.shuffle(train_clips_list)
    overlap_clips = set(train_clips_list[:OVERLAP_CLIPS])

    # 剩余未选中的视频
    remaining_clips = set(all_clip_nums) - train_clips
    # 测试集 = 重叠视频 + 剩余视频中选 (TEST_CLIPS_COUNT - OVERLAP_CLIPS) 个
    remaining_clips_list = list(remaining_clips)
    random.shuffle(remaining_clips_list)
    additional_test_clips = set(remaining_clips_list[:TEST_CLIPS_COUNT - OVERLAP_CLIPS])

    test_clips = overlap_clips | additional_test_clips

    # 验证数量
    print(f"\n训练集视频数: {len(train_clips)} (目标 {TRAIN_CLIPS_COUNT})")
    print(f"测试集视频数: {len(test_clips)} (目标 {TEST_CLIPS_COUNT})")
    print(f"重叠视频数: {len(overlap_clips)} (目标 {OVERLAP_CLIPS})")

    # 构建文件对（图片 + 对应掩码）
    train_pairs = []
    test_pairs = []
    missing_masks = []

    for img_path in all_images:
        clip_num = get_clip_num(img_path.name)
        mask_path = all_masks.get(img_path.stem)

        if mask_path is None:
            missing_masks.append(img_path.name)
            continue

        if clip_num in train_clips:
            train_pairs.append((img_path, mask_path))
        if clip_num in test_clips:
            test_pairs.append((img_path, mask_path))

    print(f"\n有效图片统计:")
    print(f"训练集: {len(train_pairs)} 张图片")
    print(f"测试集: {len(test_pairs)} 张图片")
    if missing_masks:
        print(f"警告: {len(missing_masks)} 张图片找不到掩码")

    # 输出重叠视频的具体编号
    print(f"\n重叠的视频编号: {sorted(overlap_clips)}")

    # 清理并创建输出目录
    if output_root.exists():
        print(f"\n清理旧目录: {output_root}")
        shutil.rmtree(output_root)

    # 复制文件
    print("\n开始复制文件...")
    copy_files_with_progress(train_pairs, output_root, 'training', '复制训练集')
    copy_files_with_progress(test_pairs, output_root, 'testing', '复制测试集')

    # 保存划分信息
    info = f"""# Surgery Dataset Split Information
# 划分方式: 随机划分
# 随机种子: {RANDOM_SEED}
# 训练集视频数: {len(train_clips)} (目标 {TRAIN_CLIPS_COUNT})
# 测试集视频数: {len(test_clips)} (目标 {TEST_CLIPS_COUNT})
# 重叠视频数: {len(overlap_clips)} (目标 {OVERLAP_CLIPS})
# 训练集图片数: {len(train_pairs)}
# 测试集图片数: {len(test_pairs)}

## 训练集视频列表
{train_clips}

## 测试集视频列表
{test_clips}

## 重叠视频列表
{overlap_clips}
"""
    with open(output_root / 'split_info.txt', 'w') as f:
        f.write(info)

    print(f"\n✅ 完成！输出目录: {output_root}")
    print(f"训练集: {len(train_pairs)} 张图片")
    print(f"测试集: {len(test_pairs)} 张图片")
    print(f"重叠视频: {sorted(overlap_clips)}")


if __name__ == '__main__':
    main()