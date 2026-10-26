# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from pathlib import Path
import copy
from tqdm import tqdm

# ===== 配置参数 =====
data_dir = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "data"))
num_classes = 8  # 修改为8类
batch_size = 32
epochs = 100
learning_rate = 0.0003
weight_decay = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("🌼 Top 8 菊花品种训练")
print("=" * 60)
print(f"使用设备: {device}")
print(f"数据集路径: {data_dir}")
print(f"类别数量: {num_classes}")
print(f"批次大小: {batch_size}")
print(f"训练轮数: {epochs}")
print("=" * 60)

# ===== 数据增强 =====
train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ===== 加载数据集 =====
train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'val'),
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'test'),
    transform=val_transform
)

if len(train_dataset.classes) != num_classes:
    raise ValueError('Expected exactly eight training classes.')
if not (train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx):
    raise ValueError('Class folders must match across train, val and test.')

# 获取类别名称
class_names = train_dataset.classes
print(f"\n📁 训练集: {len(train_dataset)} 张图片")
print(f"📁 验证集: {len(val_dataset)} 张图片")
print(f"📁 测试集: {len(test_dataset)} 张图片")
print(f"🏷️ 类别: {class_names}")

# ===== 创建数据加载器 =====
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ===== 创建EfficientNet-B3模型 =====
def create_efficientnet():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # 冻结特征提取层
    for param in model.features.parameters():
        param.requires_grad = False
    
    # 解冻最后几层
    for param in model.features[-5:].parameters():
        param.requires_grad = True
    
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

model = create_efficientnet().to(device)

# ===== 损失函数和优化器 =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': learning_rate * 0.1},
    {'params': model.classifier.parameters(), 'lr': learning_rate}
], weight_decay=weight_decay)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ===== 早停设置 =====
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience = 15
counter = 0

# 记录训练过程
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\n🚀 开始训练...")
print("=" * 60)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.item())
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc.item())
    
    print(f'\nEpoch {epoch+1}/{epochs}')
    print(f'  Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'class_names': class_names
        }, 'best_top8.pth')
        print(f'  ✅ 保存最佳模型，验证准确率: {val_acc:.4f}')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'  ⏹️ 早停！{patience}轮没有提升')
            break
    
    scheduler.step()
    print()

print("=" * 60)
print(f"🎉 训练完成！最佳验证准确率: {best_val_acc:.4f}")
print("最佳模型已保存: best_top8.pth")

# 测试集评估
model.load_state_dict(best_model_wts)
model.eval()
test_corrects = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='测试集评估'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_corrects.double() / len(test_dataset)
print(f"\n📊 测试集准确率: {test_acc:.4f}")
