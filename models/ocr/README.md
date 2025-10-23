# 车牌OCR识别模型

基于CRNN（卷积循环神经网络）的车牌号码识别模型，支持中国车牌的自动识别。

## 项目结构

```
models/
├── ocr/                       # OCR模型代码目录
│   ├── data_preprocessor.py   # 数据预处理模块
│   ├── crnn_model.py          # CRNN模型定义
│   ├── train_model.py         # 模型训练模块
│   ├── inference.py           # 推理检测模块
│   ├── main.py                # 主程序入口
│   ├── requirements.txt       # Python依赖
│   ├── train.sh              # 训练脚本
│   └── README.md             # 说明文档
├── plate_ocr_model.pt        # 训练好的OCR模型文件
└── runs/ocr/                 # 训练日志和结果
```

## 快速开始

### 1. 安装依赖

```bash
cd models/ocr
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
# 使用Python
python main.py --mode all

# 或使用脚本
./train.sh
```

### 3. 分步运行

```bash
# 仅数据预处理
python main.py --mode preprocess

# 仅模型训练
python main.py --mode train

# 仅推理测试
python main.py --mode inference --image path/to/image.jpg

# 批量识别
python main.py --mode inference --image_dir path/to/images/

# 模型测试
python main.py --mode test
```

## 数据格式

模型支持标准车牌OCR数据集格式，数据应存放在：

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

标签文件格式示例：
```
datasets/train/images/000000.jpg,浙YQL551
datasets/train/images/000001.jpg,黑Q49CFS
datasets/train/images/000002.jpg,桂LY7C51
```

## 模型架构

### CRNN模型

- **CNN特征提取器**: 5层卷积网络提取图像特征
- **RNN序列建模**: 双向LSTM处理序列特征
- **CTC损失函数**: 处理变长序列识别

### 技术特点

- 支持任意长度车牌号码识别
- 端到端训练，无需字符分割
- 轻量级模型，适合部署
- 高准确率，支持中文字符

## 模型配置

### 训练参数

- **输入尺寸**: 32x128 (高度x宽度)
- **训练轮次**: 30 epochs
- **批量大小**: 16
- **学习率**: 0.001 (自适应调整)
- **优化器**: Adam
- **损失函数**: CTC Loss

### 性能指标

- **识别准确率**: >95% (在测试集上)
- **推理速度**: <50ms/张 (GPU)
- **模型大小**: ~10MB

## API使用

### 初始化识别器

```python
from inference import LicensePlateOCR

ocr = LicensePlateOCR("models/plate_ocr_model.pt")
```

### 单张图像识别

```python
result = ocr.recognize_single_image("path/to/image.jpg")
print(result)  # 输出: "浙YQL551"
```

### 批量识别

```python
results = ocr.recognize_batch_images("path/to/images/", output_dir="results/")
for result in results:
    print(f"{result['image_file']}: {result['plate_text']}")
```

## 输出结果

识别结果包含：

```python
{
    'image_file': '000000.jpg',
    'plate_text': '浙YQL551',      # 识别出的车牌号码
    'success': True                 # 识别是否成功
}
```

## 自定义训练

### 修改模型配置

编辑 `train_model.py` 中的训练参数：

```python
training_config = {
    'epochs': 50,           # 训练轮次
    'batch_size': 32,       # 批量大小
    'learning_rate': 0.001, # 学习率
    # ... 其他参数
}
```

### 使用不同数据集

修改 `data_preprocessor.py` 中的数据路径：

```python
preprocessor = DataPreprocessor(data_root="path/to/your/data")
```

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 使用清华镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **内存不足**
   - 减小批量大小：`batch_size=8`
   - 减小输入尺寸：修改 `img_height` 和 `img_width`

3. **训练不收敛**
   - 检查数据标注质量
   - 调整学习率：`learning_rate=0.0001`
   - 增加训练轮次：`epochs=100`

### 性能优化

- **GPU加速**: 确保安装CUDA版本的PyTorch
- **批量推理**: 使用 `recognize_batch_images` 进行批量处理
- **模型量化**: 训练后可使用模型量化减小文件大小

## 扩展功能

### 集成车牌检测

可与车牌检测模型结合使用：

```python
# 先检测车牌位置
from plate_detection import detect_plates
plate_boxes = detect_plates(image)

# 再识别车牌号码
for box in plate_boxes:
    plate_image = crop_plate(image, box)
    plate_text = ocr.recognize_single_image(plate_image)
    print(f"车牌号码: {plate_text}")
```

### 实时识别

支持实时视频流识别：

```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    # 检测车牌
    plate_boxes = detect_plates(frame)
    
    for box in plate_boxes:
        plate_image = crop_plate(frame, box)
        plate_text = ocr.recognize_single_image(plate_image)
        
        # 在图像上显示结果
        cv2.putText(frame, plate_text, (box[0], box[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('License Plate Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。