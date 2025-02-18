import torch
from torch.utils.data import DataLoader, Dataset
from .dataset import ChemicalReactionDataLoader
import numpy as np
import random
import copy

class DataArrangement:
    def __init__(self, args, split_type="random", split_args=None,collate_func='random'):
        """
        初始化 DataArrangement 类
        :param args: 存储所有参数的字典
        :param split_type: 数据分割的方式 ("random" 或 "OOD")
        :param split_args: 分割的参数（如果是 "random"，需要给出训练、验证、测试的比例）
        """
        self.args = args
        self.split_type = split_type
        self.split_args = split_args if split_args else []
        self.collate_func = collate_func # pair,normal
        # 载入数据并进行分割
        self.fingerprints, self.labels = self.load_data()
        self.train_set, self.valid_set, self.test_set = self.split_data()

    def load_data(self):
        """
        从 ChemicalReactionDataLoader 获取数据
        :return: fingerprints 和 labels
        """
        print("Loading dataset...")
        # 假设从 ChemicalReactionDataLoader 获取数据的代码如下：
        loader = ChemicalReactionDataLoader(self.args['dataset_path'], 
                                             self.args['encoding_type'], 
                                             self.args['encoding_params'], 
                                             self.args['force_reencoding'])
        fingerprints, labels = loader.get_data(data_type="for_train")
        print(f"Dataset loaded with {len(fingerprints)} samples.")
        return fingerprints, labels

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
        total_samples = len(self.fingerprints)
        train_size = int(self.split_args[0] * total_samples)
        valid_size = int(self.split_args[1] * total_samples)
        test_size = total_samples - train_size - valid_size

        indices = list(range(total_samples))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        valid_indices = indices[train_size: train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

        train_set = (self._create_subset(train_indices), [self.labels[i] for i in train_indices])
        valid_set = (self._create_subset(valid_indices), [self.labels[i] for i in valid_indices])
        test_set = (self._create_subset(test_indices), [self.labels[i] for i in test_indices])

        return train_set, valid_set, test_set

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

        train_set = (self._create_subset(train_indices), [self.labels[i] for i in train_indices])
        valid_set = (self._create_subset(valid_indices), [self.labels[i] for i in valid_indices])
        test_set = (self._create_subset(test_indices), [self.labels[i] for i in test_indices])

        return train_set, valid_set, test_set

    def _create_subset(self, indices):
        """
        根据索引生成数据的子集
        :param indices: 索引列表
        :return: 对应的数据子集
        """
        subset = [self.fingerprints[i] for i in indices]
        return torch.tensor(subset)

    def collate_normal(self, batch):
        """
        默认的 collate function，直接返回 tensor 和 label
        :param batch: 当前 batch 的数据
        :return: { 'fingerprint': tensor, 'labels': label}
        """
        fingerprints, labels = zip(*batch)
        return {"fingerprint": torch.stack(fingerprints, dim=0), "labels": torch.tensor(labels)}

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
