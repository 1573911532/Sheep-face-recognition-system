import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
pretrained_weights = 'mobilefacenet0.pth'
model.load_state_dict(torch.load(pretrained_weights))
model.eval()

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载特征库
features = np.load('goat_features.npy')
labels = np.load('goat_labels.npy')

# 新图片路径
new_image_path = 'runs/test-15.jpg'  # 替换为新图片的实际路径

# 读取新图片
new_image = Image.open(new_image_path).convert('RGB')
# 图像预处理
new_image = transform(new_image).unsqueeze(0)

# 提取新图片的特征
with torch.no_grad():
    new_feature = model(new_image).squeeze().cpu().numpy()

# 计算新特征与特征库中所有特征的余弦相似度
similarities = cosine_similarity([new_feature], features)[0]

# 找到相似度最高的特征的索引
max_index = np.argmax(similarities)

# 获取对应的羊的标签
predicted_goat_id = labels[max_index]

# 输出识别结果
print(f"新图片中的羊预测编号为: goat{predicted_goat_id}，相似度为: {similarities[max_index]:.4f}")