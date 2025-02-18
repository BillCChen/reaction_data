import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef, R2Score
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np

class ModelRegression(pl.LightningModule):
    def __init__(self, args):
        """
        初始化多层感知机模型。
        :param args: 包含模型参数的字典，包括网络层数、激活函数、dropout、优化器等信息。
        """
        super(ModelRegression, self).__init__()
        
        self.args = args
        
        # 模型组件
        self.model = self.create_model()

        # 初始化评估指标字典
        self.eval_metrics = {
            'train_loss': np.nan,
            'valid_loss': np.nan,
            'r2': np.nan,
            'pearson_r': np.nan
        }

        # 打印模型相关信息
        print(f"Model initialized with the following parameters:\n{self.args}")
        
    def create_model(self):
        """
        根据 args 创建多层感知机模型。
        :return: nn.Module 类型的神经网络模型。
        """
        layers = []
        input_dim = self.args['input_dim']
        
        # 构建隐藏层
        for i in range(self.args['num_layers']):
            layers.append(nn.Linear(input_dim, self.args['hidden_dim']))
            if self.args.get('use_normalization', False):
                layers.append(nn.BatchNorm1d(self.args['hidden_dim']))
            layers.append(self.get_activation_function())
            if self.args['dropout'] > 0:
                layers.append(nn.Dropout(self.args['dropout']))
            input_dim = self.args['hidden_dim']

        # 最后输出层
        layers.append(nn.Linear(input_dim, 1))  # 输出维度为 1（回归问题）
        
        model = nn.Sequential(*layers)
        return model

    def get_activation_function(self):
        """
        获取激活函数，默认为 ReLU。
        :return: 激活函数
        """
        activation = self.args.get('activation', 'relu')
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入数据
        :return: 模型输出
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        :return: 优化器和学习率调度器
        """
        optimizer = optim.Adam(self.parameters(), lr=self.args.get('lr', 0.001))
        
        # 学习率调度器
        scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        """
        训练步骤
        :param batch: 当前 batch 数据
        :param batch_idx: 当前 batch 的索引
        :return: 训练损失
        """
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        
        # 更新训练损失
        self.eval_metrics['train_loss'] = loss.item()
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        :param batch: 当前 batch 数据
        :param batch_idx: 当前 batch 的索引
        :return: 验证损失和预测值
        """
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        
        # 更新评估指标
        self.eval_metrics['valid_loss'] = loss.item()
        self.eval_metrics['r2'] = R2Score()(y_hat.squeeze(), y).item()
        self.eval_metrics['pearson_r'] = PearsonCorrCoef()(y_hat.squeeze(), y).item()
        
        return loss

    def on_epoch_end(self):
        """
        每个 epoch 结束时打印当前的训练和验证损失，以及其他评估指标
        """
        print(f"Epoch {self.current_epoch} -- "
              f"Train Loss: {self.eval_metrics['train_loss']:.4f}, "
              f"Validation Loss: {self.eval_metrics['valid_loss']:.4f}, "
              f"R2: {self.eval_metrics['r2']:.4f}, "
              f"Pearson R: {self.eval_metrics['pearson_r']:.4f}")

    def get_args(self, with_metrics=False):
        """
        返回模型参数，若指定返回评估指标。
        :param with_metrics: 是否返回评估指标
        :return: args 字典或包含参数和评估指标的字典
        """
        if with_metrics:
            return {'args': self.args, 'metrics': self.eval_metrics}
        return self.args

    def return_train_parts(self):
        """
        返回模型训练所需的组件，如优化器，学习率调度器等。
        :return: 包含优化器、学习率调度器的字典
        """
        return {
            'optimizer': optim.Adam(self.parameters(), lr=self.args.get('lr', 0.001)),
            'lr_scheduler': StepLR(optimizer, step_size=10, gamma=0.7),
            'early_stopping': self.args.get('early_stopping', None)
        }

# 示例 args 设置
args = {
    'input_dim': 128,        # 输入维度
    'hidden_dim': 64,        # 隐藏层维度
    'num_layers': 3,         # 网络层数
    'activation': 'relu',    # 激活函数
    'dropout': 0.3,          # Dropout 比例
    'lr': 0.001,             # 学习率
    'batch_size': 32,        # 批量大小
    'early_stopping': 'valid_loss',  # 提前停止指标
    'use_normalization': True,   # 是否使用归一化
}

# 创建模型实例
model = ModelRegression(args)

