#!/bin/bash

# YOLOv8车牌识别模型训练脚本

echo "=== YOLOv8车牌识别模型训练 ==="

# 设置Python路径
PYTHON_PATH="python"

# 检查依赖
echo "检查Python依赖..."
$PYTHON_PATH -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 创建目录结构
echo "创建目录结构..."
mkdir -p ../../datasets/plates
mkdir -p ../../runs/detect
mkdir -p ../../results

# 数据预处理
echo "=== 数据预处理 ==="
$PYTHON_PATH data_preprocessor.py

# 检查数据预处理是否成功
if [ ! -f "../../datasets/plates/data.yaml" ]; then
    echo "错误: 数据预处理失败，data.yaml文件未生成"
    exit 1
fi

# 模型训练
echo "=== 模型训练 ==="
$PYTHON_PATH train_model.py

# 检查模型是否生成
if [ -f "../../plate_detection_model.pt" ]; then
    echo "模型训练成功! 模型文件: ../../plate_detection_model.pt"
else
    echo "警告: 模型文件未生成，请检查训练过程"
fi

# 运行推理测试
echo "=== 推理测试 ==="
$PYTHON_PATH inference.py

echo "=== 训练完成 ==="
echo "模型文件: ../../plate_detection_model.pt"
echo "训练日志: ../../runs/detect/train/"
echo "测试结果: ../../results/"

# 显示模型大小
if [ -f "../../plate_detection_model.pt" ]; then
    MODEL_SIZE=$(du -h "../../plate_detection_model.pt" | cut -f1)
    echo "模型大小: $MODEL_SIZE"
fi