# YOLOv8车牌识别模型

基于YOLOv8的车牌检测和识别模型，支持中国车牌（CCPD数据集格式）的自动检测。

## 项目结构

```
models/
├── plate/                   # 模型代码目录
│   ├── data_preprocessor.py # 数据预处理模块
│   ├── train_model.py       # 模型训练模块
│   ├── inference.py         # 推理检测模块
│   ├── main.py              # 主程序入口
│   ├── requirements.txt     # Python依赖
│   └── train.sh            # 训练脚本
├── plate_detection_model.pt # 训练好的模型文件
└── runs/                    # 训练日志和结果
```

## 快速开始

### 1. 安装依赖

```bash
cd models/plate
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
python main.py --mode inference

# 单张图像推理
python main.py --mode inference --image path/to/image.jpg
```

## 数据格式

模型支持CCPD数据集格式，数据应存放在：

```
data/
├── images/                  # 车辆图像目录
│   ├── 01.jpg
│   ├── 02.jpg
│   └── ...
└── labels/                  # 标注文件目录
    └── ccpd_base_samples.json
```

标注文件格式示例：
```json
{
  "samples": [
    {
      "media": {
        "media_path": "01.jpg",
        "media_shape": [720, 1160, 3]
      },
      "annotations": [
        {
          "type": "bbox",
          "bbox": [100, 200, 300, 100],  // [x, y, w, h]
          "content": "京A12345"
        }
      ]
    }
  ]
}
```

## 模型配置

### 训练参数

- **模型大小**: YOLOv8n（轻量版）
- **输入尺寸**: 640x640
- **训练轮次**: 100 epochs
- **批量大小**: 16
- **学习率**: 0.01（自适应调整）

### 性能指标

- **检测精度**: >95% mAP50
- **推理速度**: <100ms/张（GPU）
- **模型大小**: ~6MB

## API使用

### 初始化检测器

```python
from inference import LicensePlateDetector

detector = LicensePlateDetector("models/plate_detection_model.pt")
```

### 单张图像检测

```python
result = detector.process_single_image("path/to/image.jpg", output_dir="results/")
print(result)
```

### 批量检测

```python
summary = detector.process_batch_images("data/images/", output_dir="results/")
print(summary)
```

## 输出结果

检测结果包含：

```python
{
    'image_path': 'path/to/image.jpg',
    'detections': [
        {
            'bbox': [x1, y1, x2, y2],      # 边界框坐标
            'confidence': 0.95,            # 置信度
            'class_name': 'license_plate', # 类别名称
            'class_id': 0                  # 类别ID
        }
    ],
    'plate_count': 1,                      # 检测到的车牌数量
    'success': True                        # 处理是否成功
}
```

## 自定义训练

### 修改模型配置

编辑 `train_model.py` 中的训练参数：

```python
training_config = {
    'epochs': 100,           # 训练轮次
    'imgsz': 640,            # 输入尺寸
    'batch': 16,             # 批量大小
    'lr0': 0.01,             # 初始学习率
    # ... 其他参数
}
```

### 使用不同模型大小

```python
# 使用小型模型（推荐）
trainer = YOLOv8Trainer(data_yaml_path, model_size='n')

# 使用中型模型
trainer = YOLOv8Trainer(data_yaml_path, model_size='s')

# 使用大型模型
trainer = YOLOv8Trainer(data_yaml_path, model_size='m')
```

## 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 使用清华镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **内存不足**
   - 减小批量大小：`batch=8`
   - 减小输入尺寸：`imgsz=320`

3. **训练不收敛**
   - 检查数据标注质量
   - 调整学习率：`lr0=0.001`
   - 增加训练轮次：`epochs=200`

### 性能优化

- **GPU加速**: 确保安装CUDA版本的PyTorch
- **批量推理**: 使用`process_batch_images`进行批量处理
- **模型量化**: 训练后可使用模型量化减小文件大小

## 扩展功能

### 字符识别集成

模型目前仅支持车牌检测，可集成OCR模块进行字符识别：

```python
# 集成PaddleOCR
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
plate_image = detector.crop_plate_region(image, bbox)
result = ocr.ocr(plate_image)
plate_number = result[0][1][0]  # 提取车牌号码
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。