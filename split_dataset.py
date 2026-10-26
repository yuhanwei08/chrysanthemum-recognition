# -*- coding: utf-8 -*-
import os
from pathlib import Path
import shutil
import random

dataset_dir = os.environ.get("RAW_DATA_DIR", str(Path(__file__).resolve().parent / "chrysanthemum_dataset"))
target_dir = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "data"))
train_ratio = 0.7
val_ratio = 0.15

if Path(target_dir).exists() and any(Path(target_dir).rglob('*')):
    raise SystemExit('Output data folder is not empty. Choose a fresh DATA_DIR to avoid overlapping splits.')

random.seed(42)

# 修改为最终的8个品种
varieties = ["乒乓菊", "勋章菊", "墨菊", "波斯菊", "金丝皇菊", "金鸡菊", "雏菊", "黑眼菊"]

for variety in varieties:
    if not (Path(dataset_dir) / variety).is_dir():
        raise SystemExit(f'Missing source class folder: {variety}')

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

for variety in varieties:
    src_dir = os.path.join(dataset_dir, variety)
    if not os.path.isdir(src_dir):
        print(f"警告：{variety} 文件夹不存在，跳过")
        continue
    images = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    
    for split_name, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        dst_dir = os.path.join(target_dir, split_name, variety)
        os.makedirs(dst_dir, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    print(f"{variety} 划分完成")

print("数据集划分完毕！")