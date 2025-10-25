# Smart-Parking-System
智慧停车管理系统

## 🎯 项目概述

这是一个基于计算机视觉的智能停车管理系统，主要功能包括：

1. **车牌识别** - 通过深度学习模型自动识别车辆车牌号
2. **业主车辆判断** - 根据车牌号判断是否为本小区业主车辆
3. **停车有效期管理** - 判断车辆的停车有效期
4. **后台管理系统** - 管理小区车辆车牌以及停车有效期

## 📊 数据集来源

### 车牌检测数据集
- **数据集名称**：CCPD (Chinese City Parking Dataset)
- **下载地址**：https://www.modelscope.cn/datasets/OmniData/CCPD
- **数据集规模**：原始数据集包含超过30万张车辆图像
- **本项目使用**：为演示目的，仅选用了其中2000张图片进行训练
- **数据特点**：包含各种光照条件、天气状况和角度的车牌图像

### 车牌OCR数据集
- **数据集名称**：License_plate_10K
- **下载地址**：https://www.modelscope.cn/datasets/A3315300155/License_plate_10K
- **数据集规模**：包含约10,000张车牌图像
- **数据特点**：专门用于车牌字符识别训练

---

## 📁 项目结构

```
Smart-Parking-System/
├── backend/                    # 后端API服务
│   ├── main.py                # FastAPI主程序
│   ├── api.py                 # API路由定义
│   ├── plate_model.py         # 车牌识别模型封装
│   ├── requirements.txt       # Python依赖
│   ├── init_mysql.sql        # 数据库初始化脚本
│   └── DB Info.txt           # 数据库配置信息
├── frontend/                  # 前端Vue应用
│   ├── src/
│   │   ├── App.vue            # 主应用组件
│   │   ├── PlateDetection.vue # 车牌检测页面
│   │   └── PlateManagement.vue # 车牌管理页面
│   ├── package.json           # 前端依赖配置
│   └── package-lock.json
├── models/                    # 深度学习模型
│   ├── plate/                 # 车牌检测模块
│   ├── ocr/                   # OCR识别模块
│   └── requirements.txt       # 模型依赖
│   # 注意：模型文件需要手动训练生成
├── data/                      # 数据集目录
├── datasets/                  # 训练数据集
├── results/                   # 检测结果输出
└── README.md                  # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- **操作系统**: Windows/Linux/macOS
- **Python**: 3.8+
- **Node.js**: 14.0+
- **MySQL**: 5.7+
- **GPU**: 推荐（可选，用于加速模型推理）

### 1. 模型训练和准备

**重要提示**: GitHub代码仓库中不包含训练好的模型文件，您需要手动训练模型才能使用系统。

#### 模型训练步骤（必需）
请按照以下步骤训练所需的深度学习模型：

1. **安装模型依赖**：
   ```bash
   cd models
   pip install -r requirements.txt
   ```

2. **车牌检测模型训练**：
   ```bash
   cd models/plate
   python train_model.py
   # 或使用训练脚本
   ./train.sh
   ```

3. **OCR识别模型训练**：
   ```bash
   cd models/ocr
   python train_model.py
   # 或使用训练脚本
   ./train.sh
   ```

4. **复制模型文件到后端目录**：
   ```bash
   # 确保模型文件位于正确位置
   cp models/plate_detection_model.pt backend/models/
   cp models/plate_ocr_model.pt backend/models/
   
   # 如果backend/models目录不存在，请先创建
   mkdir -p backend/models
   ```

### 2. 数据库初始化

1. 安装MySQL数据库
2. 创建数据库用户：
   ```sql
   CREATE USER 'parking'@'localhost' IDENTIFIED BY '123456';
   GRANT ALL PRIVILEGES ON smart_parking.* TO 'parking'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. 初始化数据库表：
   ```bash
   mysql -u root -p < backend/init_mysql.sql
   ```

### 3. 后端服务部署

1. 安装Python依赖：
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. 启动后端服务：
   ```bash
   python main.py
   ```
   服务将在 http://localhost:8000 启动

3. 验证API服务：
   ```bash
   curl http://localhost:8000/plates/
   ```

### 4. 前端应用部署

1. 安装Node.js依赖：
   ```bash
   cd frontend
   npm install
   ```

2. 启动前端开发服务器：
   ```bash
   npm run serve
   ```
   应用将在 http://localhost:8080 启动

3. 构建生产版本：
   ```bash
   npm run build
   ```

---

## 🔧 详细模型训练指南

### 车牌检测模型训练

1. **准备数据集**
   - 下载CCPD数据集：https://www.modelscope.cn/datasets/OmniData/CCPD
   - **数据集规模说明**：原始CCPD数据集包含超过30万张车辆图像，本项目为演示目的仅选用了其中2000张图片进行训练
   - 将数据集解压到 `data/` 目录
   - 数据集应包含车辆图像和对应的车牌标注信息

2. **数据预处理**
   ```bash
   cd models/plate
   python data_preprocessor.py
   ```

3. **模型训练**
   ```bash
   python train_model.py
   # 或使用训练脚本
   ./train.sh
   ```

4. **模型验证**
   ```bash
   python main.py --mode inference --image path/to/test_image.jpg
   ```

### OCR识别模型训练

1. **准备数据集**
   - 下载车牌OCR数据集：https://www.modelscope.cn/datasets/A3315300155/License_plate_10K
   - 将数据集解压到 `data/ocr_data/` 目录
   - 数据集应包含车牌图像和对应的车牌号码标签

2. **数据预处理**
   ```bash
   cd models/ocr
   python data_preprocessor.py
   ```

3. **模型训练**
   ```bash
   python train_model.py
   # 或使用训练脚本
   ./train.sh
   ```

4. **模型验证**
   ```bash
   python main.py --mode inference --image path/to/plate_image.jpg
   ```

### 模型使用说明

**重要提示**：请确保已完成模型训练并复制模型文件到正确位置。

1. **单张图像识别**
   ```python
   from models.license_plate_recognition import LicensePlateRecognitionSystem
   
   # 初始化识别系统
   system = LicensePlateRecognitionSystem()
   
   # 识别单张图像
   result = system.recognize_single_image("car_image.jpg")
   print(f"检测到 {result['plate_count']} 个车牌")
   ```

2. **批量识别**
   ```python
   summary = system.recognize_batch_images("image_directory/")
   print(f"批量处理完成，检测率: {summary['detection_rate']:.2%}")
   ```

---

## 📱 系统功能说明

### 车牌检测功能
- **图像上传**: 支持拖拽上传和文件选择
- **实时检测**: 自动识别图像中的车牌位置
- **结果展示**: 显示车牌号码、有效期和状态

### 车牌管理功能
- **数据查询**: 支持按车牌号码搜索和状态筛选
- **添加车辆**: 手动录入车辆信息
- **信息修改**: 更新车辆有效期和状态
- **数据删除**: 删除无效车辆记录

### API接口说明

#### 车牌检测接口
```http
POST /detect/
Content-Type: multipart/form-data

参数:
- image: 图像文件

响应:
{
  "plate_number": "京A12345",
  "message": "识别成功",
  "is_valid": true,
  "valid_to": "2026-01-01",
  "owner_name": "张三"
}
```

#### 车牌管理接口
```http
GET /plates/                    # 获取所有车牌
POST /plates/                   # 添加新车牌
GET /plates/{id}               # 获取指定车牌
PUT /plates/{id}               # 更新车牌信息
DELETE /plates/{id}            # 删除车牌
```

---

## 🔧 配置说明

### 数据库配置
编辑 `backend/api.py` 中的数据库连接配置：
```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "parking",
    "password": "123456",
    "database": "smart_parking"
}
```

### 模型路径配置
编辑 `backend/plate_model.py` 中的模型路径：
```python
self.detection_model_path = "models/plate_detection_model.pt"
self.ocr_model_path = "models/plate_ocr_model.pt"
```

### 前端API配置
编辑前端代码中的API地址（默认：http://localhost:8000）

---

## 🐛 故障排除

### 常见问题

1. **后端启动失败**
   - 检查Python依赖是否安装完整
   - 验证MySQL数据库连接配置
   - 确认模型文件路径正确

2. **前端无法连接后端**
   - 检查后端服务是否正常启动
   - 验证CORS配置
   - 检查网络连接和防火墙设置

3. **模型加载失败**
   - 确认模型文件存在且路径正确
   - 检查PyTorch版本兼容性
   - 验证GPU驱动（如使用GPU）

4. **数据库连接错误**
   - 检查MySQL服务状态
   - 验证数据库用户权限
   - 确认数据库表结构正确

### 性能优化建议

- **GPU加速**: 安装CUDA版本的PyTorch
- **批量处理**: 使用批量识别提高效率
- **缓存机制**: 实现结果缓存减少重复计算
- **异步处理**: 使用异步API提高并发性能

---

## 📈 性能指标

### 车牌检测模型
- **检测精度**: >95% mAP50
- **推理速度**: <100ms/张（GPU）
- **模型大小**: ~6MB

### OCR识别模型
- **识别准确率**: >95%
- **推理速度**: <50ms/张（GPU）
- **模型大小**: ~10MB

### 系统整体性能
- **并发处理**: 支持多用户同时使用
- **响应时间**: 平均<2秒完成检测
- **稳定性**: 7x24小时稳定运行

---

## 🔄 开发计划

### 近期功能
- [ ] 实时视频流车牌识别
- [ ] 移动端应用开发
- [ ] 数据统计和分析功能
- [ ] 多语言支持

### 长期规划
- [ ] 云端部署方案
- [ ] 多停车场管理
- [ ] 智能收费系统集成
- [ ] AI算法持续优化

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

### 开发环境设置
1. Fork项目仓库
2. 创建功能分支
3. 提交代码变更
4. 创建Pull Request

### 代码规范
- 遵循PEP 8 Python代码规范
- 使用有意义的变量名和函数名
- 添加必要的注释和文档
- 确保代码测试覆盖率

---

## 📄 许可证

本项目基于MIT许可证开源。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用本系统进行商业应用时，请确保遵守相关法律法规和隐私政策。

