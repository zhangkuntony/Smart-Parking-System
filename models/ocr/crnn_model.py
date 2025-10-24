import torch
import torch.nn as nn
import os

class CRNN(nn.Module):
    """CRNN模型用于车牌识别"""
    
    def __init__(self, num_classes, img_height=32, img_width=128, hidden_size=256, num_layers=2):
        """
        初始化CRNN模型
        
        Args:
            num_classes: 字符类别数量
            img_height: 图像高度
            img_width: 图像宽度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
        """
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            # 输入: [batch, 1, 32, 128]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 16, 64]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 8, 32]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [256, 4, 32]
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [512, 2, 32]
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [512, 1, 31]
        )
        
        # 计算CNN输出特征图大小
        self._init_cnn_output_size()
        
        # RNN序列建模
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def _init_cnn_output_size(self):
        """计算CNN输出特征图大小"""
        try:
            # 模拟前向传播计算输出大小
            x = torch.zeros(1, 1, self.img_height, self.img_width)
            x = self.cnn(x)
            self.cnn_output_size = x.size(1) * x.size(2)  # channels * height
            self.rnn_input_length = x.size(3)  # width
        except Exception as e:
            # 如果计算失败，使用默认值
            print(f"警告: 无法计算CNN输出尺寸，使用默认值: {e}")
            self.cnn_output_size = 512  # 默认值
            self.rnn_input_length = 31   # 默认值
        
    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        
        # CNN特征提取
        x = self.cnn(x)  # [batch, 512, 1, 31]
        
        # 重塑为序列格式 [batch, sequence_length, feature_size]
        x = x.squeeze(2)  # [batch, 512, 31]
        x = x.permute(0, 2, 1)  # [batch, 31, 512]
        
        # RNN序列建模
        x, _ = self.rnn(x)  # [batch, 31, hidden_size * 2]
        
        # 输出层
        x = self.fc(x)  # [batch, 31, num_classes]
        x = x.permute(1, 0, 2)  # [31, batch, num_classes] for CTC loss
        
        return x
    
    def predict(self, x):
        """预测车牌号码"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # 使用argmax获取每个时间步的预测
            predictions = torch.argmax(outputs, dim=2)  # [31, batch]
            predictions = predictions.permute(1, 0)  # [batch, 31]
            
            return predictions

class LicensePlateRecognizer:
    """车牌识别器"""
    
    def __init__(self, model_path=None, device=None):
        """
        初始化车牌识别器
        
        Args:
            model_path: 模型文件路径
            device: 设备 (cpu/cuda)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 字符映射（训练时更新）
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.num_classes = 0
        
        # 如果有预训练模型，先读取类别数再创建模型
        if model_path and os.path.exists(model_path):
            # 先读取模型文件获取类别数
            checkpoint = torch.load(model_path, map_location=self.device)
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            
            # 修复：使用保存的num_classes，而不是重新计算
            if 'num_classes' in checkpoint:
                self.num_classes = checkpoint['num_classes']
            else:
                # 如果没有保存num_classes，则使用字符映射的长度
                self.num_classes = len(self.char_to_idx)
            
            # 创建正确大小的模型
            self.model = CRNN(self.num_classes).to(self.device)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功，字符集大小: {self.num_classes}")
            print(f"字符映射: {len(self.char_to_idx)} 个字符")
        else:
            # 如果没有预训练模型，初始化一个最小模型
            self.num_classes = 68  # 常见中文字符+数字+字母
            self.model = CRNN(self.num_classes).to(self.device)
    
    def load_model(self, model_path):
        """加载模型（现在在初始化时已经加载）"""
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            self.num_classes = len(self.char_to_idx)
            
            # 重新创建正确大小的模型
            self.model = CRNN(self.num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功，字符集大小: {self.num_classes}")
        else:
            print("模型文件不存在，使用默认模型")
    
    def save_model(self, model_path, optimizer=None, epoch=None, loss=None):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'num_classes': self.num_classes
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, model_path)
        print(f"模型已保存到: {model_path}")
    
    def set_character_mapping(self, char_to_idx, idx_to_char):
        """设置字符映射"""
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        # 总类别数 = 字符数 + 1（空白符）
        self.num_classes = len(char_to_idx) + 1
        
        # 更新模型输出层
        old_fc = self.model.fc
        self.model.fc = nn.Linear(old_fc.in_features, self.num_classes).to(self.device)
    
    def decode_predictions(self, predictions):
        """解码预测结果"""
        decoded_texts = []
        
        for pred in predictions:
            # CTC解码：移除重复字符和空白符
            text = ''
            last_char = -1
            for char_idx in pred:
                # 确保索引在有效范围内
                char_idx = int(char_idx)
                if char_idx < 0 or char_idx >= len(self.idx_to_char):
                    # 索引超出范围，跳过
                    continue
                    
                if char_idx != last_char:
                    # 空白符索引为0，跳过空白符
                    if char_idx != 0 and char_idx in self.idx_to_char:
                        text += self.idx_to_char[char_idx]
                    last_char = char_idx
            decoded_texts.append(text)
        
        return decoded_texts
    
    def recognize_single_image(self, image_tensor):
        """识别单张图像"""
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        
        image_tensor = image_tensor.to(self.device)
        predictions = self.model.predict(image_tensor)
        texts = self.decode_predictions(predictions)
        
        return texts[0] if texts else ""

if __name__ == "__main__":
    # 测试模型
    model = CRNN(num_classes=68)
    x = torch.randn(2, 1, 32, 128)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试识别器
    recognizer = LicensePlateRecognizer()
    pred = recognizer.model.predict(x)
    print(f"预测形状: {pred.shape}")