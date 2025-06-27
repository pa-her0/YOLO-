import os
import shutil
import random

# 原始数据路径
base_dir = './'
dataset_dir = os.path.join(base_dir, 'output_dataset')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 新的输出路径（不在 DATA1.0 下）
output_dir = os.path.join(base_dir, 'dataset')
train_images_dir = os.path.join(output_dir, 'images', 'train')
val_images_dir = os.path.join(output_dir, 'images', 'val')
train_labels_dir = os.path.join(output_dir, 'labels', 'train')
val_labels_dir = os.path.join(output_dir, 'labels', 'val')

# 创建新目录
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 收集所有图像文件
image_files = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    and os.path.isfile(os.path.join(images_dir, f))
]

# 打乱顺序
random.shuffle(image_files)

# 划分比例
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# 复制训练集
for img in train_images:
    base_name, ext = os.path.splitext(img)
    label = base_name + '.txt'

    # 图像复制
    shutil.copy(os.path.join(images_dir, img), os.path.join(train_images_dir, img))

    # 标签复制
    label_src = os.path.join(labels_dir, label)
    label_dst = os.path.join(train_labels_dir, label)
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)
    else:
        print(f"[警告] 缺失标签文件: {label_src}")

# 复制验证集
for img in val_images:
    base_name, ext = os.path.splitext(img)
    label = base_name + '.txt'

    # 图像复制
    shutil.copy(os.path.join(images_dir, img), os.path.join(val_images_dir, img))

    # 标签复制
    label_src = os.path.join(labels_dir, label)
    label_dst = os.path.join(val_labels_dir, label)
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)
    else:
        print(f"[警告] 缺失标签文件: {label_src}")

print(f"✅ 数据集划分完成，训练集：{len(train_images)}，验证集：{len(val_images)}")
print(f"✅ 输出路径：{output_dir}")