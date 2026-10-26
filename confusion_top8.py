# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt  # 必须在使用 plt 之前导入
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from pathlib import Path

# 设置中文字体（放在导入之后）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 配置 =====
data_dir = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent / "data"))
model_path = "best_top8.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['乒乓菊', '勋章菊', '墨菊', '波斯菊', '金丝皇菊', '金鸡菊', '雏菊', '黑眼菊']

# ===== 数据预处理 =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ===== 加载测试集 =====
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"测试集图片数: {len(test_dataset)}")

# ===== 创建模型 =====
def create_model(num_classes=8):
    model = models.efficientnet_b3(weights=None)
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

model = create_model(8)
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
classes = checkpoint.get('class_names', classes)
if test_dataset.classes != classes:
    raise ValueError('Test class order does not match checkpoint class_names.')
labels_en = ['Pingpang Ju', 'Xunzhang Ju', 'Mo Ju', 'Bosi Ju', 'Jinsi Huangju', 'Jinji Ju', 'Chu Ju', 'Heiyan Ju']
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()
print("模型加载成功")

# ===== 预测 =====
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(f'Test samples: {len(all_labels)}; accuracy: {sum(a == b for a, b in zip(all_labels, all_preds)) / len(all_labels):.4%}')

# ===== 混淆矩阵 =====
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_en, yticklabels=labels_en,
            annot_kws={'size': 12})
plt.title('Eight-class recognition confusion matrix', fontsize=16)
plt.xlabel('Predicted class', fontsize=14)
plt.ylabel('True class', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_top8.png', dpi=300)
print("混淆矩阵已保存为 confusion_matrix_top8.png")

# ===== 分类报告（文本） =====
report = classification_report(all_labels, all_preds, labels=list(range(len(classes))), target_names=labels_en, zero_division=0)
print("\n分类报告：\n", report)

# 保存分类报告
with open('classification_report_top8.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("分类报告已保存为 classification_report_top8.txt")