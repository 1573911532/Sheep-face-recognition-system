# 羊脸识别系统

基于 MobileFaceNet 和 YOLOv8 的羊脸识别系统，支持图片和视频的实时检测与识别。

## 功能特点

- 🖼️ 图片检测：支持单张图片的羊脸检测和识别
- 📹 视频检测：支持视频文件的实时检测和识别
- 📸 摄像头检测：支持实时摄像头检测和识别
- 🎯 高精度识别：结合 YOLOv8 目标检测和 MobileFaceNet 特征提取
- 📊 实时显示：实时显示检测结果和识别信息
- 🗄️ 特征库管理：支持添加和管理羊脸特征库

## 系统要求

- Python 3.8+
- CUDA 支持（推荐用于 GPU 加速）
- 至少 4GB 内存
- 支持 OpenCV 的摄像头（用于实时检测）

## 安装步骤

1. 克隆项目：
```bash
git clone [项目地址]
cd [项目目录]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
- 将 MobileFaceNet 预训练模型 `mobilefacenet.pth` 放在项目根目录
- 将 YOLOv8 预训练模型 `yolov8_model.pt` 放在项目根目录

4. 创建必要的目录：
```bash
mkdir -p static/uploads
mkdir -p static/yolov8_results
mkdir -p static/yolov8_crops
mkdir -p static/feature_images
mkdir -p static/temp_videos
```

## 使用方法

1. 启动应用：
```bash
python app.py
```

2. 访问系统：
- 打开浏览器，访问 `http://localhost:5000`

3. 功能使用：
   - 图片检测：选择"图片检测"模式，上传图片进行检测
   - 视频检测：选择"视频文件检测"模式，上传视频文件进行检测
   - 摄像头检测：选择"摄像头实时检测"模式，点击"开始摄像头检测"
   - 特征库管理：点击"查看特征库中的羊数据"进行管理

## 项目结构

```
├── app/
│   ├── templates/
│   │   └── index.html      # 前端页面
│   └── app.py              # 后端应用
├── static/
│   ├── uploads/            # 上传文件存储
│   ├── yolov8_results/     # YOLOv8 检测结果
│   ├── yolov8_crops/       # 裁剪的羊脸图片
│   ├── feature_images/     # 特征库图片
│   └── temp_videos/        # 临时视频文件
├── mobilefacenet.pth       # MobileFaceNet 预训练模型
├── yolov8_model.pt         # YOLOv8 预训练模型
├── goat_features.npy       # 羊脸特征库
├── goat_labels.npy         # 羊脸标签库
└── requirements.txt        # 项目依赖
```

## 技术说明

- 使用 YOLOv8 进行羊脸检测
- 使用 MobileFaceNet 进行特征提取和识别
- 使用 Flask 构建 Web 应用
- 使用 OpenCV 处理图像和视频
- 使用 PyTorch 进行深度学习推理

## 注意事项

1. 确保摄像头权限已开启（用于实时检测）
2. 视频文件大小建议不超过 100MB
3. 特征库图片建议使用清晰的羊脸正面照片
4. 识别相似度阈值设置为 0.95，可根据需要调整

## 常见问题

1. 如果遇到摄像头无法打开：
   - 检查摄像头权限
   - 确保没有其他程序占用摄像头

2. 如果视频检测不流畅：
   - 降低视频分辨率
   - 减少检测帧率

3. 如果识别准确率不高：
   - 调整相似度阈值
   - 增加特征库样本数量
   - 确保输入图片质量

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

[MIT License](LICENSE)



