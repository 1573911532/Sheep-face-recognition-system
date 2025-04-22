import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# 定义 MobileFaceNet 模型
class MobileFaceNet(torch.nn.Module):
    def __init__(self, embedding_size=128):
        super(MobileFaceNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            # 更多层...
        )
        feature_dim = 64 * 56 * 56  # 需根据实际情况调整
        self.fc = torch.nn.Linear(feature_dim, embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 加载预训练模型
model = MobileFaceNet()
pretrained_weights = 'mobilefacenet.pth'
model.load_state_dict(torch.load(pretrained_weights))
model.eval()

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 数据集路径
data_folder = 'target'

# 存储特征和标签
features = []
labels = []

# 遍历数据集
for goat_folder in os.listdir(data_folder):
    if goat_folder.startswith('goat'):
        goat_id = int(goat_folder.replace('goat', ''))
        goat_path = os.path.join(data_folder, goat_folder)
        for img_name in os.listdir(goat_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(goat_path, img_name)
                # 读取图像
                img = Image.open(img_path).convert('RGB')
                # 图像预处理
                img = transform(img).unsqueeze(0)
                # 提取特征
                with torch.no_grad():
                    feature = model(img)
                features.append(feature.squeeze().cpu().numpy())
                labels.append(goat_id)

# 转换为 NumPy 数组
features = np.array(features)
labels = np.array(labels)

# 保存特征库
np.save('goat_features.npy', features)
np.save('goat_labels.npy', labels)

print("特征库建立完成，特征保存为 goat_features.npy，标签保存为 goat_labels.npy")