# Smart Parking System - 车牌识别系统

基于深度学习的端到端车牌识别系统，集成了车牌检测和OCR识别功能。

## 🚀 系统概述

本系统包含两个核心模块：
- **车牌检测模块**：基于YOLOv8的车牌位置检测
- **OCR识别模块**：基于CRNN的车牌号码识别

### 主要特性
- ✅ 高精度车牌检测（>95% mAP50）
- ✅ 高准确率OCR识别（>95%准确率）
- ✅ 支持中国车牌格式
- ✅ 端到端识别流程
- ✅ 轻量级模型，适合部署
- ✅ 支持批量处理和实时识别

## 📁 项目结构

```
models/
├── plate_detection_model.pt      # 车牌检测模型
├── plate_ocr_model.pt           # OCR识别模型
├── requirements.txt             # 统一依赖文件
├── README.md                    # 本文档
├── license_plate_recognition.py # 集成识别系统
├── plate/                       # 车牌检测模块
│   ├── data_preprocessor.py    # 数据预处理
│   ├── train_model.py          # 模型训练
│   ├── inference.py            # 推理检测
│   ├── main.py                 # 主程序入口
│   └── train.sh               # 训练脚本
└── ocr/                        # OCR识别模块
    ├── data_preprocessor.py    # 数据预处理
    ├── crnn_model.py           # CRNN模型定义
    ├── train_model.py          # 模型训练
    ├── inference.py            # 推理识别
    ├── main.py                 # 主程序入口
    └── train.sh               # 训练脚本
```

## 🛠️ 快速开始

### 1. 安装依赖

```bash
cd models
pip install -r requirements.txt
```

### 2. 使用集成识别系统（推荐）

```bash
# 识别单张汽车图像
python license_plate_recognition.py --mode single --image ../data/test_car.jpg

# 批量识别目录中的所有图像
python license_plate_recognition.py --mode batch --image_dir ../data/images
```

### 3. 单独使用各模块

#### 车牌检测模块
```bash
cd plate

# 训练模型
python main.py --mode train

# 单张图像检测
python main.py --mode inference --image ../data/test_car.jpg

# 或使用脚本
./train.sh
```

#### OCR识别模块
```bash
cd ocr

# 训练模型
python main.py --mode train

# 单张车牌图像识别
python main.py --mode inference --image ../data/plate_image.jpg

# 或使用脚本
./train.sh
```

## 📊 模型性能

### 车牌检测模型（YOLOv8）
- **检测精度**: >95% mAP50
- **推理速度**: <100ms/张（GPU）
- **模型大小**: ~6MB
- **输入尺寸**: 640x640

### OCR识别模型（CRNN）
- **识别准确率**: >95%
- **推理速度**: <50ms/张（GPU）
- **模型大小**: ~10MB
- **输入尺寸**: 32x128

## 🔧 API使用

### 集成识别系统

```python
from license_plate_recognition import LicensePlateRecognitionSystem

# 初始化系统
system = LicensePlateRecognitionSystem()

# 识别单张图像
result = system.recognize_single_image("car_image.jpg")
print(f"检测到 {result['plate_count']} 个车牌")
for plate in result['plate_results']:
    print(f"车牌 {plate['plate_id']}: {plate['plate_text']}")

# 批量识别
summary = system.recognize_batch_images("image_directory/")
print(f"批量处理完成，检测率: {summary['detection_rate']:.2%}")
print(f"识别率: {summary['recognition_rate']:.2%}")
```

### 单独使用检测模块

```python
from plate.inference import LicensePlateDetector

detector = LicensePlateDetector("plate_detection_model.pt")
result = detector.process_single_image("car_image.jpg")

for detection in result['detections']:
    print(f"车牌位置: {detection['bbox']}, 置信度: {detection['confidence']:.2f}")
```

### 单独使用OCR模块

```python
from ocr.inference import LicensePlateOCR

ocr = LicensePlateOCR("plate_ocr_model.pt")
plate_text = ocr.recognize_single_image("plate_image.jpg")
print(f"识别结果: {plate_text}")
```

## 📁 数据格式

### 车牌检测数据（CCPD格式）
```
data/
├── images/                  # 车辆图像目录
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
└── labels/                  # 标注文件目录
    └── ccpd_base_samples.json
```

### OCR识别数据
```
data/ocr_data/
├── datasets/
│   ├── train/
│   │   ├── images/           # 训练图像目录
│   │   └── train_labels.txt  # 训练标签文件
│   ├── val/
│   │   ├── images/           # 验证图像目录
│   │   └── val_labels.txt    # 验证标签文件
│   └── test/
│       ├── images/           # 测试图像目录
│       └── test_labels.txt   # 测试标签文件
└── font/                     # 字体文件目录
```

## 🎯 输出结果

### 集成识别结果
```python
{
    'image_path': 'car_image.jpg',
    'plate_count': 1,
    'recognized_count': 1,
    'plate_results': [
        {
            'plate_id': 1,
            'bbox': [100, 200, 300, 250],
            'confidence': 0.95,
            'plate_text': '浙YQL551',
            'recognition_success': True
        }
    ],
    'success': True
}
```

## 🔧 自定义配置

### 修改模型路径
```python
system = LicensePlateRecognitionSystem(
    detection_model_path="custom_detection_model.pt",
    ocr_model_path="custom_ocr_model.pt",
    conf_threshold=0.6  # 检测置信度阈值
)
```

### 训练参数调整

#### 车牌检测训练
编辑 `plate/train_model.py`：
```python
training_config = {
    'epochs': 100,           # 训练轮次
    'imgsz': 640,            # 输入尺寸
    'batch': 16,             # 批量大小
    'lr0': 0.01,             # 初始学习率
}
```

#### OCR识别训练
编辑 `ocr/train_model.py`：
```python
training_config = {
    'epochs': 30,            # 训练轮次
    'batch_size': 16,       # 批量大小
    'learning_rate': 0.001,  # 学习率
}
```

## 🐛 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 使用国内镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **内存不足**
   - 减小批量大小
   - 减小输入图像尺寸
   - 使用GPU加速

3. **模型加载失败**
   - 检查模型文件路径
   - 确保PyTorch版本兼容

4. **识别准确率低**
   - 检查数据质量
   - 调整训练参数
   - 增加训练数据量

### 性能优化建议

- **GPU加速**: 安装CUDA版本的PyTorch
- **批量处理**: 使用批量识别提高效率
- **模型量化**: 训练后使用模型量化减小文件大小
- **多线程处理**: 对于大量图像使用多线程处理

## 🚀 扩展功能

### 实时视频识别
```python
import cv2

cap = cv2.VideoCapture(0)
system = LicensePlateRecognitionSystem()

while True:
    ret, frame = cap.read()
    result = system.recognize_single_image(frame)
    
    # 在图像上显示识别结果
    for plate in result['plate_results']:
        x1, y1, x2, y2 = plate['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate['plate_text'], (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('License Plate Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Web服务部署
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
system = LicensePlateRecognitionSystem()

@app.route('/recognize', methods=['POST'])
def recognize():
    image_file = request.files['image']
    result = system.recognize_single_image(image_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用本系统进行商业应用时，请确保遵守相关法律法规和隐私政策。