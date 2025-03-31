import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

from model.model import LitModel as regression_model
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

@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    
    version = "FULL"
    print(f"Data Version: {version}")
    pl.seed_everything(cfg.seed)
    # torch.set_float32_matmul_precision('medium')
    torch.set_float32_matmul_precision('high') 
    # Load your data_dict here
    if version == "MINI":
        # data_dict = pickle.load(open("/root/public_data/20250303_ChORISO_train_test_reactions_fps_labels.pkl", "rb"))
        data_dict = pickle.load(open("/root/public_data/20250303_ChORISO_train_test_reactions_fps_labels_mini.pkl", "rb"))
        # Convert to tensors
        train_fps = torch.stack(data_dict["train_fps"], dim=0)
        test_fps = torch.stack(data_dict["test_fps"], dim=0) 
        train_labels = torch.tensor(data_dict["train_labels"], dtype=torch.float32).view(-1, 1) - torch.tensor([0.7])
        test_labels = torch.tensor(data_dict["test_labels"], dtype=torch.float32).view(-1, 1) - torch.tensor([0.7])
    elif version == "FULL":
        import h5py

        with h5py.File("/root/public_data/ChORISO/data.h5", "r") as f:
            train_fps = torch.tensor(f["train_fps"][:], dtype=torch.float32)
            test_fps = torch.tensor(f["test_fps"][:], dtype=torch.float32)
            train_labels = torch.tensor(f["train_labels"][:], dtype=torch.float32).view(-1, 1)
            # 正态化
            train_labels = (train_labels - train_labels.mean()) / train_labels.std()
            test_labels = torch.tensor(f["test_labels"][:], dtype=torch.float32).view(-1, 1)
            # 正态化
            test_labels = (test_labels - test_labels.mean()) / test_labels.std()
    else:
        raise ValueError(f"Invalid version of data: {version}")
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
    # model = regression_model(cfg.model,cfg.optimizer,cfg.scheduler)
    model = regression_model()
    print(f"Input Dim: {cfg.model.input_dim}, Hidden Dim: {cfg.model.hidden_dim}, Num Layers: {cfg.model.num_layers}, Output Dim: {cfg.model.output_dim}, LR: {cfg.optimizer.lr}, Use BN: {cfg.model.use_bn}")
    from hydra.utils import instantiate

# 自动实例化 callbacks
    # trainer = Trainer(
    # max_epochs=cfg.trainer.max_epochs,
    # min_epochs=cfg.trainer.min_epochs,
    # devices=cfg.trainer.devices,
    # accelerator=cfg.trainer.accelerator,
    # logger=cfg.trainer.logger,
    # check_val_every_n_epoch=cfg.trainer.val_check_interval,
    # num_sanity_val_steps=10,
    # # progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
    # callbacks=[instantiate(x) for x in cfg.trainer.callbacks]
    # # +[epoch_progress_bar],
    # )
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, dm)
    print("Training finished")
    trainer.test(model, datamodule=dm)
    
    # Save final config
    OmegaConf.save(config=cfg, f=os.path.join(os.getcwd(), "config.yaml"))

if __name__ == "__main__":
    main()
    #source regression_env/bin/activate