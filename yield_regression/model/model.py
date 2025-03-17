import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics import PearsonCorrCoef, R2Score
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


import matplotlib.pyplot as plt
import os
from pathlib import Path
class ModelRegression(pl.LightningModule):
    def __init__(self, args):
        """
        初始化多层感知机模型。
        :param args: 包含模型参数的字典，包括网络层数、激活函数、dropout、优化器等信息。
        """
        super(ModelRegression, self).__init__()
        
        self.args = args
        # 为了防止 args 中缺少某些参数，这里进行一些默认设置
        # self.args.setdefault('input_dim', 2048)
        # self.args.setdefault('hidden_dim', 64)
        # self.args.setdefault('num_layers', 3)
        # self.args.setdefault('activation', 'relu')
        # self.args.setdefault('dropout', 0.3)
        # self.args.setdefault('lr', 0.001)
        # self.args.setdefault('batch_size', 32)
        # self.args.setdefault('early_stopping', None)
        # self.args.setdefault('use_normalization', False)

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
        x, y = batch['fingerprint'], batch['labels']    
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
        x, y = batch['fingerprint'], batch['labels']  
        y_hat = self(x)
        loss = F.mse_loss(y_hat.squeeze(), y)
        
        # 更新评估指标
        self.eval_metrics['valid_loss'] = loss.item()
        self.eval_metrics['r2'] = R2Score()(y_hat.squeeze(), y).item()
        self.eval_metrics['pearson_r'] = PearsonCorrCoef()(y_hat.squeeze(), y).item()
        
        return loss

    def on_epoch_end(self,epoch):
        """
        每个 epoch 结束时打印当前的训练和验证损失，以及其他评估指标
        """
        print(f"Epoch {epoch} -- "
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
        optimizer = optim.Adam(self.parameters(), lr=self.args.get('lr', 0.001))
        lr_sceduler = StepLR(optimizer, step_size=10, gamma=0.7)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_sceduler,
            'early_stopping': self.args.get('early_stopping', None)
        }
    
# 示例 args 设置
# args = {
#     'input_dim': 128,        # 输入维度
#     'hidden_dim': 64,        # 隐藏层维度
#     'num_layers': 3,         # 网络层数
#     'activation': 'relu',    # 激活函数
#     'dropout': 0.3,          # Dropout 比例
#     'lr': 0.001,             # 学习率
#     'batch_size': 32,        # 批量大小
#     'early_stopping': 'valid_loss',  # 提前停止指标
#     'use_normalization': True,   # 是否使用归一化
# }

# # 创建模型实例
# model = ModelRegression(args)
class LitModel(pl.LightningModule):
    def __init__(self, input_dim=2048, hidden_dim=1024, num_layers=8, output_dim=1, 
                 lr=1e-3, use_bn=True,dropout_ratio=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        layers = []
        # Input layer
        layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_ratio))
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.test_preds = []
        self.test_targets = []
        self.valid_container_preds = []
        self.valid_container_targets = []
    def forward(self, x):
        return self.model(x)
    def poisson_loss(self,pred, target):
        """
        计算泊松损失函数
        :param pred: 模型的预测值，形状为 (batch_size,)
        :param target: 真实值，形状为 (batch_size,)
        :return: 泊松损失值
        """
        # 避免对数运算中的数值不稳定问题
        eps = 1e-10
        pred = torch.clamp(pred, min=eps)
        # 计算泊松损失
        loss = torch.mean(pred - target * torch.log(pred))
        return loss
    def list_wise_loss(self,target, prediction):
        """
        计算列表式排序损失
        :param target: 目标值，形状可能为 (batch_size,) 或更高维度
        :param prediction: 预测值，形状可能为 (batch_size,) 或更高维度
        :return: 列表式排序损失
        """
        # 确保 target 和 prediction 是 1D 张量
        # 计算所有样本对的目标值差值和预测值差值
        target_diff = target - target.t()  # 形状为 (batch_size, batch_size)
        prediction_diff = prediction - prediction.t()  # 形状为 (batch_size, batch_size)

        # 避免对角线元素（自身与自身的比较）对损失产生影响
        mask = torch.ones_like(target_diff, dtype=torch.bool)
        # 手动将对角线元素置零
        for i in range(mask.size(0)):
            mask[i, i] = 0

        # 只考虑上三角部分（避免重复计算）
        upper_triangular_mask = torch.triu(mask, diagonal=1)

        # 计算 Pairwise Ranking Loss
        pairwise_loss = torch.log(1 + torch.exp(-target_diff * prediction_diff))

        # 提取上三角部分的损失
        valid_losses = pairwise_loss[upper_triangular_mask]

        # 计算平均损失
        loss = valid_losses.mean()

        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        loss = self.poisson_loss(y_hat, y)
        # loss = self.list_wise_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        self.valid_container_preds.append(y_hat)
        self.valid_container_targets.append(y)
        return {"preds": y_hat, "targets": y}

    def on_validation_epoch_end(self):
        # 获取在 validation_step 中记录的所有预测值和目标值
        preds = torch.cat([x.cpu() for x in self.valid_container_preds])
        targets = torch.cat([x.cpu() for x in self.valid_container_targets])
        pearson = PearsonCorrCoef()(preds.squeeze(), targets.squeeze())
        spearman = SpearmanCorrCoef()(preds.squeeze(), targets.squeeze())
        r_2  = 1 - F.mse_loss(preds, targets) / torch.var(targets)
        combined = (pearson + spearman) / 2
        # 清空 self.valid_container_preds 和 self.valid_container_targets
        self.valid_container_preds = []
        self.valid_container_targets = []
        self.log("val_p", pearson)
        self.log("val_s", spearman)
        self.log("val_p_s", combined, prog_bar=True)
        self.log("val_r2", r_2, prog_bar=True)
        return combined

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_preds.append(y_hat)
        self.test_targets.append(y)
        return {"preds": y_hat, "targets": y}

    def on_test_epoch_end(self):
        # 获取在 test_step 中记录的所有预测值和目标值
        preds = torch.cat([x.cpu() for x in self.test_preds])
        targets = torch.cat([x.cpu() for x in self.test_targets])
        self.test_preds = []
        self.test_targets = []
        pearson = PearsonCorrCoef()(preds.squeeze(), targets.squeeze())
        spearman = SpearmanCorrCoef()(preds.squeeze(), targets.squeeze())
        r2 = 1 - F.mse_loss(preds, targets) / torch.var(targets)
        self.log("test_pearson", pearson)
        self.log("test_spearman", spearman)
        self.log("test_r2", r2)
        # Plot and save
        plt.figure(figsize=(10, 6))
        plt.scatter(targets.cpu().numpy(), preds.detach().cpu().numpy(), alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"Test Set Predictions\nPearson: {pearson:.4f}, Spearman: {spearman:.4f} R2: {r2:.4f}")
        plt.savefig(Path(os.getcwd()) / "test_scatter.png")
        plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.regression import R2Score
import matplotlib.pyplot as plt
from pathlib import Path
import os
import logging

class VaeModel(pl.LightningModule):
    def __init__(self, model_args, optimizer_args, scheduler_args):
        super().__init__()
        # 保存所有参数
        self.save_hyperparameters()
        
        # 初始化模型组件
        self.init_model()
        
        # 容器用于验证和测试
        self.valid_container_preds = []
        self.valid_container_targets = []
        self.test_preds = []
        self.test_targets = []
        
        # 初始化指标计算器
        self.pearson = PearsonCorrCoef()
        self.spearman = SpearmanCorrCoef()
        self.r2_score = R2Score()

    def init_model(self):
        """Initialize model components based on model_args."""
        # VAE Encoder
        self.vae_encoder = self._build_encoder(
            input_dim=self.hparams.model_args.input_dim,
            hidden_dim=self.hparams.model_args.hidden_dim,
            num_layers=self.hparams.model_args.num_layers
        )
        
        # VAE Decoder
        self.vae_decoder = self._build_decoder(
            input_dim=self.hparams.model_args.input_dim,
            hidden_dim=self.hparams.model_args.hidden_dim,
            num_layers=self.hparams.model_args.num_layers
        )
        
        # MLP for the first half of latent space
        self.mlp = self._build_mlp(
            input_dim=self.hparams.model_args.hidden_dim,
            output_dim=self.hparams.model_args.output_dim,
            num_layers=self.hparams.model_args.num_layers
        )
    def get_norm_function(self,args_use_bn,dim):
        """
        获取归一化函数
        :param args_use_bn: 是否使用 BatchNorm
        :return: 归一化函数
        """
        if args_use_bn:
            return nn.BatchNorm1d(dim)
        return nn.Identity()
    def _build_encoder(self, input_dim, hidden_dim, num_layers):
        """Build the VAE encoder with dynamic hidden layers."""
        layers = []
        in_features = input_dim
        out_features = input_dim // 2
        
        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, out_features))
            if self.hparams.model_args.use_bn:
                layers.append(self.get_norm_function(self.hparams.model_args.use_bn,out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.model_args.dropout_ratio))
            
            # Update dimensions for next layer
            in_features = out_features
            out_features = max(out_features // 2, hidden_dim)  # Ensure final layer outputs latent_dim * 2
        
        # Final layer to output mean and log variance
        layers.append(nn.Linear(in_features, hidden_dim))
        
        return nn.Sequential(*layers)

    def _build_decoder(self, input_dim,hidden_dim, num_layers):
        """Build the VAE decoder with dynamic hidden layers."""
        layers = []
        in_features = hidden_dim // 3 * 2
        out_features = hidden_dim // 3 * 2 * 2
        
        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, out_features))
            if self.hparams.model_args.use_bn:
                layers.append(self.get_norm_function(self.hparams.model_args.use_bn,out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.model_args.dropout_ratio))
            
            # Update dimensions for next layer
            in_features = out_features
            out_features = min(out_features * 2, input_dim)
        layers.append(nn.Linear(in_features, input_dim))
        
        return nn.Sequential(*layers)

    def _build_mlp(self, input_dim, output_dim, num_layers):
        """Build the MLP with dynamic hidden layers."""
        layers = []
        in_features = input_dim // 3 
        out_features = input_dim // 3 // 2
        
        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, out_features))
            if self.hparams.model_args.use_bn:
                layers.append(self.get_norm_function(self.hparams.model_args.use_bn,out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.model_args.dropout_ratio))
            
            # Update dimensions for next layer
            in_features = out_features
            out_features = max(out_features // 2, output_dim)
        
        # Final layer to output prediction
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

    def vae_encoder_forward(self, x):
        """Forward pass for VAE encoder."""
        h = self.vae_encoder(x)
        info,mu, log_var = torch.chunk(h, 3, dim=-1)  # Split into mean and log variance
        return info,mu, log_var

    def vae_decoder_forward(self, z):
        """Forward pass for VAE decoder."""
        return self.vae_decoder(z)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruction_loss(self, x, x_recon):
        """Compute reconstruction loss (MSE)."""
        return F.mse_loss(x_recon, x, reduction='mean')

    def KL_loss_half(self, mu, log_var):
        """Compute KL divergence loss for the second half of latent dimensions."""
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        # 不只计算每一个元素的正态分布的最大似然估计，还计算整个分布的最大似然估计，使得协方差矩阵更接近单位矩阵
        
        return kl_loss.mean()
    def covariance_loss(self, z):
        """Compute the covariance regularization loss."""
        # z: (batch_size, latent_dim)
        batch_size, latent_dim = z.shape
        
        # 计算均值
        z_mean = z.mean(dim=0, keepdim=True)  # (1, latent_dim)
        
        # 计算协方差矩阵
        z_centered = z - z_mean  # (batch_size, latent_dim)
        cov_matrix = torch.matmul(z_centered.T, z_centered) / (batch_size - 1)  # (latent_dim, latent_dim)
        
        # 计算单位矩阵
        identity_matrix = torch.eye(latent_dim, device=z.device)  # (latent_dim, latent_dim)
        
        # 计算协方差矩阵与单位矩阵的差异（Frobenius 范数）
        cov_loss = torch.norm(cov_matrix - identity_matrix, p="fro")  # 标量
        
        return cov_loss
    def total_loss(self, mu, log_var, z, lambda_cov=1.0):
        """Compute the total loss, including KL divergence and covariance regularization."""
        # KL 散度损失
        kl_loss = self.KL_loss_half(mu, log_var)
        
        # 协方差正则化损失
        cov_loss = self.covariance_loss(z)
        
        # 总损失
        total_loss = kl_loss + lambda_cov * cov_loss
    
        # return total_loss, kl_loss, cov_loss
        return total_loss
    def forward(self, x):
        """Full forward pass for the model."""
        # VAE Encoder
        info,mu, log_var = self.vae_encoder_forward(x)
        z = self.reparameterize(mu, log_var)
        info_z = torch.cat([info, z], dim=-1)
        # VAE Decoder
        x_recon = self.vae_decoder_forward(info_z)
        
        # MLP for the first half of latent space
        z_first_half = z[:, :32]
        output = self.mlp(z_first_half)
        
        return x_recon, output, mu, log_var,z

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        x_recon, output, mu, log_var,z = self(x)
        
        # Compute losses
        recon_loss = self.reconstruction_loss(x, x_recon)
        # kl_loss = self.KL_loss_half(mu, log_var)
        kl_loss = self.total_loss(mu, log_var, z)
        mse_loss = F.mse_loss(output, y)
        
        # Total loss
        total_loss = 0.0001 * recon_loss + 0.001 * kl_loss + mse_loss * 1.0
        
        # Log losses
        self.log("train_recon_loss", recon_loss, prog_bar=True,logger=True)
        self.log("train_kl_loss", kl_loss, prog_bar=True,logger=True)
        self.log("train_mse_loss", mse_loss, prog_bar=True,logger=True)
        self.log("train_total_loss", total_loss, prog_bar=True,logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        x_recon, output, mu, log_var,z = self(x)
        
        # Compute losses
        recon_loss = self.reconstruction_loss(x, x_recon)
        # kl_loss = self.KL_loss_half(mu, log_var)
        kl_loss = self.total_loss(mu, log_var, z)
        mse_loss = F.mse_loss(output, y)
        
        # Log losses
        self.log("val_recon_loss", recon_loss, prog_bar=True,logger=True)
        self.log("val_kl_loss", kl_loss, prog_bar=True,logger=True)
        self.log("val_mse_loss", mse_loss, prog_bar=True,logger=True)
        
        # Store predictions and targets for epoch-end calculation
        self.valid_container_preds.append(output)
        self.valid_container_targets.append(y)
        
        return {"preds": output, "targets": y}

    def on_validation_epoch_end(self):
        # 获取所有验证集的预测值和目标值
        preds = torch.cat(self.valid_container_preds)
        targets = torch.cat(self.valid_container_targets)
        
        # 计算 R² 分数
        r2 = self.r2_score(preds.squeeze(), targets.squeeze())
        pearson = self.pearson(preds.squeeze(), targets.squeeze())
        spearman = self.spearman(preds.squeeze(), targets.squeeze())
        # 清空容器
        self.valid_container_preds.clear()
        self.valid_container_targets.clear()
        
        # 记录 R² 分数
        self.log("val_r2", r2, prog_bar=True,logger=True)
        self.log("val_pearson", pearson, prog_bar=True,logger=True)
        self.log("val_spearman", spearman, prog_bar=True,logger=True)
        
        return r2

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        x_recon, output, mu, log_var,z = self(x)
        
        # Compute losses
        recon_loss = self.reconstruction_loss(x, x_recon)
        # kl_loss = self.KL_loss_half(mu, log_var)
        kl_loss = self.total_loss(mu, log_var, z)
        mse_loss = F.mse_loss(output, y)
        
        # Log losses
        self.log("test_recon_loss", recon_loss, prog_bar=True,logger=True)
        self.log("test_kl_loss", kl_loss, prog_bar=True,logger=True)
        self.log("test_mse_loss", mse_loss, prog_bar=True,logger=True)
        
        # Store predictions and targets for epoch-end calculation
        self.test_preds.append(output)
        self.test_targets.append(y)
        
        return {"preds": output, "targets": y}

    def on_test_epoch_end(self):
        # 获取所有测试集的预测值和目标值
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        
        # 计算 R² 分数
        r2 = self.r2_score(preds.squeeze(), targets.squeeze())
        pearson = self.pearson(preds.squeeze(), targets.squeeze())
        spearman = self.spearman(preds.squeeze(), targets.squeeze())
        # 清空容器
        self.test_preds.clear()
        self.test_targets.clear()
        
        self.log("test_pearson", pearson)
        self.log("test_spearman", spearman)
        # 记录 R² 分数
        self.log("test_r2", r2, prog_bar=True)
        
        # 获取 Hydra 的输出目录
        output_dir = Path(os.getcwd())  # Hydra 会自动将工作目录设置为 logs/${now:%Y-%m-%d}/${now:%H-%M-%S}
        
        # 生成文件名
        plot_filename = "test.png"
        plot_path = output_dir / plot_filename  # 完整路径为 logs/${now:%Y-%m-%d}/${now:%H-%M-%S}/test.png
        plot_path = output_dir / plot_filename
        
        # 绘制散点图并保存
        plt.figure(figsize=(10, 6))
        plt.scatter(targets.cpu().numpy(), preds.detach().cpu().numpy(), alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"Test Set Predictions\nR²: {r2:.4f}")
        plt.savefig(plot_path)
        plt.close()
        
        return r2

    def configure_optimizers(self):
        """Configure optimizer and scheduler based on optimizer_args and scheduler_args."""
        optimizer_args = self.hparams.optimizer_args
        scheduler_args = self.hparams.scheduler_args
        
        # 初始化优化器
        if optimizer_args.type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=optimizer_args.lr, 
                weight_decay=optimizer_args.get("weight_decay", 0.0)
            )
        elif optimizer_args.type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=optimizer_args.lr, 
                weight_decay=optimizer_args.get("weight_decay", 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_args.type}")
        
        # 初始化调度器
        if scheduler_args.type == "step_lr":
            scheduler = StepLR(
                optimizer, 
                step_size=scheduler_args.step_size, 
                gamma=scheduler_args.gamma
            )
        elif scheduler_args.type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, 
                patience=scheduler_args.patience, 
                factor=scheduler_args.factor
            )
        else:
            scheduler = None
        
        # 返回优化器和调度器
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_r2",  # 使用 R² 作为早停的监控指标
                },
                "gradient_clip_val": optimizer_args.get("gradient_clip_val", 0.0),
            }
        else:
            return {
                "optimizer": optimizer,
                "gradient_clip_val": optimizer_args.get("gradient_clip_val", 0.0),
            }