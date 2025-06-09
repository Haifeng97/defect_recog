import os
import random
import shutil
from pathlib import Path

# 设置随机种子保证可复现
random.seed(42)

# 原始图像和标签路径
base_path = Path("D:/projects/defect_recog/data")
image_dir = base_path / "images"
label_dir = base_path / "labels"

# 输出路径
train_img_dir = image_dir / "train"
val_img_dir = image_dir / "val"
train_lbl_dir = label_dir / "train"
val_lbl_dir = label_dir / "val"

# 创建输出目录
for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    p.mkdir(parents=True, exist_ok=True)

# 获取所有图像文件（假设为 .jpg 或 .png）
img_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
random.shuffle(img_files)

# 计算分割索引
split_idx = int(0.8 * len(img_files))
train_imgs = img_files[:split_idx]
val_imgs = img_files[split_idx:]

# 分别复制训练集
def copy_data(img_list, dest_img_dir, dest_lbl_dir):
    for img_path in img_list:
        label_name = img_path.with_suffix('.txt').name
        label_path = label_dir / label_name

        # 复制图像
        shutil.copy(img_path, dest_img_dir / img_path.name)

        # 复制对应标签
        if label_path.exists():
            shutil.copy(label_path, dest_lbl_dir / label_name)
        else:
            print(f"Warning: Label for {img_path.name} not found.")

# 执行复制
copy_data(train_imgs, train_img_dir, train_lbl_dir)
copy_data(val_imgs, val_img_dir, val_lbl_dir)

print(f"✔ Total images: {len(img_files)}")
print(f"✔ Train: {len(train_imgs)} → {train_img_dir}")
print(f"✔ Val:   {len(val_imgs)} → {val_img_dir}")
