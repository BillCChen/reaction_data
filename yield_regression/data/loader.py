import torch
from torch.utils.data import DataLoader, Dataset
from .dataset import ChemicalReactionDataset
import numpy as np
import random
import copy

class DataArrangement:
    def __init__(self, args, smiles,fingerprints, labels,split_type="random",collate_func='normal'):
        """
        初始化 DataArrangement 类
        :param args: 存储所有参数的字典
        :param split_type: 数据分割的方式 ("random" 或 "OOD")
        :param split_args: 分割的参数（如果是 "random"，需要给出训练、验证、测试的比例）
        :param collate_func: collate function 的名称
        """
        self.args = args
        self.split_type = split_type
        self.collate_func = collate_func # pair,normal
        # 载入数据并进行分割
        self.load_data(fingerprints, labels)
        
        
        self.smiles = smiles
        self.fold5 = self.split_data()

    def load_data(self,fingerprints, labels):
        """
        从 ChemicalReactionDataLoader 获取数据
        :return: fingerprints 和 labels
        """
        assert len(fingerprints) == len(labels), "fingerprints 和 labels 的长度不一致"
        assert isinstance(fingerprints,list), "fingerprints 必须是 list 类型"
        assert isinstance(labels,list), "labels 必须是 list 类型"
        assert isinstance(labels[0][0],float) , "labels[0] 必须是 float 类型"
        assert isinstance(fingerprints[0], torch.Tensor) and fingerprints[0].dtype == torch.float32, \
        "fingerprints[0] 必须是 torch.Tensor 类型且数据类型为 torch.float32"
        print("Loading dataset...")
        self.fingerprints, self.labels = fingerprints, labels
        print(f"Loaded {len(self.fingerprints)} samples.")
        print(f"Loaded {len(self.labels)} labels.")
        
        

    def split_data(self):
        """
        根据分割方式进行数据分割
        :return: train_set, valid_set, test_set
        """
        print(f"Splitting data using {self.split_type} method...")

        if self.split_type == "random":
            return self.random_split()
        elif self.split_type == "OOD":
            return self.OOD_split()
        else:
            raise ValueError(f"Unknown split type: {self.split_type}")

    def random_split(self):
        """
        使用随机方式分割数据集
        :return: 分割后的训练集、验证集、测试集
        """
        total_samples = len(self.labels)
        fold5_num = [0,int(0.2*total_samples),int(0.4*total_samples),int(0.6*total_samples),int(0.8*total_samples),int(total_samples)]
        fold_length = int(total_samples/5)
        random_index = np.random.permutation(total_samples)
        fold5_dataset = []
        for idx in range(5):
            valid_1fold_index = random_index[fold5_num[idx]:fold5_num[idx+1]]
            train_1fold_index = [i for i in range(total_samples) if i not in valid_1fold_index]
            train_dataset = ( torch.stack([self.fingerprints[i] for i in train_1fold_index ]), 
                              torch.tensor([self.labels[i] for i in train_1fold_index]),
                              [self.smiles[i] for i in train_1fold_index])
            valid_dataset = ( torch.stack([self.fingerprints[i] for i in valid_1fold_index ]),
                              torch.tensor([self.labels[i] for i in valid_1fold_index]),
                              [self.smiles[i] for i in valid_1fold_index])
            fold5_dataset.append((train_dataset, valid_dataset))
        return fold5_dataset
    def OOD_split(self):
        """
        使用 OOD 方式分割数据集
        :return: 分割后的训练集、验证集、测试集
        """
        if len(self.split_args) != len(self.fingerprints):
            raise ValueError("split_args must have the same length as the dataset.")

        train_indices = [i for i, x in enumerate(self.split_args) if x == 'train']
        valid_indices = [i for i, x in enumerate(self.split_args) if x == 'valid']
        test_indices = [i for i, x in enumerate(self.split_args) if x == 'test']

        train_set = (self._create_subset(train_indices), torch.tensor([self.labels[i] for i in train_indices]))
        valid_set = (self._create_subset(valid_indices), torch.tensor([self.labels[i] for i in valid_indices]))
        test_set = (self._create_subset(test_indices), torch.tensor([self.labels[i] for i in test_indices]))

        return train_set, valid_set, test_set


    def collate_normal(self, batch):
        """
        默认的 collate function，直接返回 tensor 和 label
        :param batch: 当前 batch 的数据
        :return: { 'fingerprint': tensor, 'labels': label}
        """
        fingerprints, labels = zip(*batch)
        return {"fingerprint": torch.stack(fingerprints), "labels": torch.tensor(labels)}

    def collate_pair(self, batch):
        """
        用于处理成对预测的 collate function
        :param batch: 当前 batch 的数据
        :return: {'fingerprint': tensor, 'labels': label}
        """
        fingerprints, labels = zip(*batch)
        fingerprints = torch.stack(fingerprints, dim=0)
        labels = torch.tensor(labels)
        
        # pair-wise subtraction (cycle)
        fingerprints = fingerprints - torch.roll(fingerprints, shifts=1, dims=0)
        labels = labels - torch.roll(labels, shifts=1, dims=0)

        return {"fingerprint": fingerprints, "labels": labels}
    def _get_collate_func(self, collate_func):
        """
        根据 collate_func 的名称返回对应的函数
        :param collate_func: collate function 的名称
        :return: collate function
        """
        if collate_func == "normal":
            return self.collate_normal
        elif collate_func == "pair":
            return self.collate_pair
        else:
            raise ValueError(f"Unknown collate function: {collate_func}")
    def get_dataloader(self, dataset_type="train"):
        """
        返回训练、验证、测试数据集的 dataloader
        :param dataset_type: "train", "valid", "test"
        :return: DataLoader 对象
        """
        batch_size = self.args.get('batch_size', 32)
        num_workers = self.args.get('num_workers', 0)
        collate_fn = self._get_collate_func(self.collate_func)
        if dataset_type == "train":
            dataset = self.train_set
            shuffle = True
        elif dataset_type == "valid":
            dataset = self.valid_set
            shuffle = False
        elif dataset_type == "test":
            dataset = self.test_set
            shuffle = False
        elif dataset_type == "full":
            dataset = (self.fingerprints, self.labels)
            shuffle = False
        else:
            raise ValueError("Unknown dataset type: {}".format(dataset_type))

        dataset = torch.utils.data.TensorDataset(dataset[0], torch.tensor(dataset[1]))
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=shuffle)

    def get_all_dataloaders(self):
        """
        一次性返回所有四个数据集的 dataloader
        :return: 四个数据集的 dataloader（train, valid, test, full）
        """
        print("Creating dataloaders for all datasets...")
        train_loader = self.get_dataloader("train")
        valid_loader = self.get_dataloader("valid")
        test_loader = self.get_dataloader("test")
        full_loader = self.get_dataloader("full")

        return train_loader, valid_loader, test_loader, full_loader
