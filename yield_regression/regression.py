import os
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pickle
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
import matplotlib.pyplot as plt
from tqdm import tqdm
class DataModule(pl.LightningDataModule):
    def __init__(self, train_fps, test_fps, train_labels, test_labels, batch_size=4096, num_workers=8, val_split=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_dataset = TensorDataset(train_fps, train_labels)
        self.test_dataset = TensorDataset(test_fps, test_labels)
        self.train_set, self.val_set = None, None

    def setup(self, stage=None):
        train_size = int((1 - self.val_split) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_set, self.val_set = random_split(
            self.train_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, 
                        shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, 
                        num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                        num_workers=self.num_workers, pin_memory=True)
# class EpochProgressBar(pl.Callback):
#     def __init__(self):
#         self.epoch_progress_bar = None

#     def on_train_start(self, trainer, pl_module):
#         # 获取最大训练轮数
#         max_epochs = trainer.max_epochs
#         # 初始化进度条
#         self.epoch_progress_bar = tqdm(total=max_epochs, desc="Training Epochs", unit="epoch",ncols=100)

#     def on_train_epoch_end(self, trainer, pl_module):
#         # 更新进度条
#         self.epoch_progress_bar.update(1)

#     def on_train_end(self, trainer, pl_module):
#         # 关闭进度条
#         self.epoch_progress_bar.close()
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
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
        combined = (pearson + spearman) / 2
        # 清空 self.valid_container_preds 和 self.valid_container_targets
        self.valid_container_preds = []
        self.valid_container_targets = []
        self.log("val_p", pearson)
        self.log("val_s", spearman)
        self.log("val_p_s", combined, prog_bar=True)
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
        pearson = PearsonCorrCoef()(preds.squeeze(), targets.squeeze())
        spearman = SpearmanCorrCoef()(preds.squeeze(), targets.squeeze())
        
        self.log("test_pearson", pearson)
        self.log("test_spearman", spearman)
        
        # Plot and save
        plt.figure(figsize=(10, 6))
        plt.scatter(targets.cpu().numpy(), preds.detach().cpu().numpy(), alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"Test Set Predictions\nPearson: {pearson:.4f}, Spearman: {spearman:.4f}")
        plt.savefig(Path(os.getcwd()) / "test_scatter.png")
        plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium') 
    # Load your data_dict here
    # data_dict = {
    #     "train_fps": [torch.randn(2048, dtype=torch.float32) for _ in range(1000)],
    #     "test_fps": [torch.randn(2048, dtype=torch.float32) for _ in range(200)],
    #     "train_labels": [torch.randn(1, dtype=torch.float32) for _ in range(1000)],
    #     "test_labels": [torch.randn(1, dtype=torch.float32) for _ in range(200)],
    # }
    # data_dict = pickle.load(open("/root/public_data_gpu8/sch_data/20250303_ChORISO_train_test_reactions_fps_labels.pkl", "rb"))
    data_dict = pickle.load(open("/root/public_data/20250303_ChORISO_train_test_reactions_fps_labels.pkl", "rb"))
    # Convert to tensors
    train_fps = torch.stack(data_dict["train_fps"], dim=0)
    test_fps = torch.stack(data_dict["test_fps"], dim=0)
    train_labels = torch.tensor(data_dict["train_labels"], dtype=torch.float32).view(-1, 1)
    test_labels = torch.tensor(data_dict["test_labels"], dtype=torch.float32).view(-1, 1)
    print(f"Train FPS: {train_fps.shape}, Train Labels: {train_labels.shape}")
    print(f"Test_FPS:{test_labels.shape}, Test_labels:{test_labels.shape}")
    # print(C)
    dm = DataModule(
        train_fps, test_fps, train_labels, test_labels,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        val_split=cfg.datamodule.val_split
    )
    print(f"Batch Size: {cfg.datamodule.batch_size}, Num Workers: {cfg.datamodule.num_workers}, Val Split: {cfg.datamodule.val_split}")
    print(f"Model: {cfg.model}")
    model = LitModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=cfg.model.output_dim,
        lr=cfg.model.lr,
        use_bn=cfg.model.use_bn,
        dropout_ratio=cfg.model.dropout
    )
    print(f"Input Dim: {cfg.model.input_dim}, Hidden Dim: {cfg.model.hidden_dim}, Num Layers: {cfg.model.num_layers}, Output Dim: {cfg.model.output_dim}, LR: {cfg.model.lr}, Use BN: {cfg.model.use_bn}")
    from hydra.utils import instantiate

# 自动实例化 callbacks
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    # epoch_progress_bar = EpochProgressBar()
    trainer = Trainer(
    max_epochs=cfg.trainer.max_epochs,
    devices=cfg.trainer.devices,
    accelerator=cfg.trainer.accelerator,
    logger=cfg.trainer.logger,
    callbacks=[instantiate(x) for x in cfg.trainer.callbacks]
    # +[epoch_progress_bar],
    )
    trainer.fit(model, dm)
    print("Training finished")
    trainer.test(model, datamodule=dm)
    
    # Save final config
    OmegaConf.save(config=cfg, f=os.path.join(os.getcwd(), "config.yaml"))

if __name__ == "__main__":
    main()
    #source regression_env/bin/activate