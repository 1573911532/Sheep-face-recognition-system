import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(MobileFaceNet, self).__init__()
        # 这里可以根据实际的 MobileFaceNet 结构进行详细定义
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 更多层...
        )
        # 假设特征维度为 64 * 56 * 56（需根据实际情况调整）
        feature_dim = 64 * 56 * 56
        self.fc = nn.Linear(feature_dim, embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GoatFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        for goat_folder in os.listdir(root_dir):
            if goat_folder.startswith('goat'):
                goat_id = int(goat_folder.replace('goat', ''))
                goat_path = os.path.join(root_dir, goat_folder)
                for img_name in os.listdir(goat_path):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(goat_path, img_name)
                        self.data.append((img_path, goat_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 创建数据集和数据加载器
    train_dataset = GoatFaceDataset(root_dir='target', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MobileFaceNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 保存模型
    torch.save(model.state_dict(), 'mobilefacenet0.pth')
    print("Model saved as mobilefacenet.pth")