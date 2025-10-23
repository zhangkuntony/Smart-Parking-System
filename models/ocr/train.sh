#!/bin/bash

# 车牌OCR模型训练脚本

echo "=== 车牌OCR识别模型训练 ==="

# 设置Python路径
PYTHON_PATH="python"

# 检查依赖
echo "检查Python依赖..."
$PYTHON_PATH -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 创建目录结构
echo "创建目录结构..."
mkdir -p ../../results/ocr_results
mkdir -p ../../runs/ocr

# 数据预处理
echo "=== 数据预处理 ==="
$PYTHON_PATH main.py --mode preprocess

# 检查数据预处理是否成功
if [ $? -ne 0 ]; then
    echo "错误: 数据预处理失败"
    exit 1
fi

# 模型训练
echo "=== 模型训练 ==="
$PYTHON_PATH main.py --mode train --epochs 30 --batch_size 16 --learning_rate 0.001

# 检查模型是否生成
if [ -f "../../plate_ocr_model.pt" ]; then
    echo "模型训练成功! 模型文件: ../../plate_ocr_model.pt"
else
    echo "警告: 模型文件未生成，请检查训练过程"
fi

# 运行测试
echo "=== 模型测试 ==="
$PYTHON_PATH main.py --mode test

# 运行推理示例
echo "=== 推理示例 ==="
$PYTHON_PATH main.py --mode inference --image_dir "../../data/ocr_data/datasets/test/images" --output_dir "../../results/ocr_results"

echo "=== 训练完成 ==="
echo "模型文件: ../../plate_ocr_model.pt"
echo "训练日志: ../../runs/ocr/"
echo "测试结果: ../../results/ocr_results/"

# 显示模型大小
if [ -f "../../plate_ocr_model.pt" ]; then
    MODEL_SIZE=$(du -h "../../plate_ocr_model.pt" | cut -f1)
    echo "模型大小: $MODEL_SIZE"
fi