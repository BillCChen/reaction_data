import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import os
from data.loader import  DataArrangement
from data.dataset import ChemicalReactionDataset

from model.model import ModelRegression
from pipeline import pipeline
from evaluater.evaluater import evaluator
from torch.utils.data import TensorDataset

if __name__ == "__main__":
    # 设置一些基本的参数
    args = {
        'dataset_path': '/root/reaction_data/S-F/SF640_A>B>C_lbl.pkl',  # 数据集路径
        'encoding_type': 'DRFP',  # 编码类型
        'input_dim': 2048,  # 输入维度
        'force_reencoding': False,  # 是否强制重新编码
        'split_type': 'random',  # 数据分割类型 ('random' 或 'OOD')
        'batch_size': 32,  # 批量大小
        'num_workers': 0,  # 工作线程数
        'max_epochs': 36,  # 最大训练轮数
        'lr': 0.001,  # 学习率
        'model_save_path': 'path_to_save_model.ckpt',  # 模型保存路径
        'predictions_save_path': 'predictions.pkl',  # 预测结果保存路径
        'gpus': None,  # GPU 设置
        'early_stopping': 'valid_loss',  # 提前停止的监控指标
    }

    # 1. 数据加载和预处理
    print("Loading dataset and preparing dataloaders...")
    Chemical_data = ChemicalReactionDataset(args,args['dataset_path'], args['encoding_type'], args['force_reencoding'])
    data_dict = Chemical_data.get_data('label','smiles',"encoding")
    smiles, labels, fingerprints = data_dict['smiles'], data_dict['label'], data_dict['encoding']
    # 使用 DataArrangement 处理数据
    dataset = DataArrangement(args, smiles,fingerprints, labels, split_type=args['split_type'], )
    # assert len(dataset.fold5[0]) == 3 # train_data,test_data,smiles
    print(f"Data loaded and prepared. {len(dataset.fold5)} folds created. Each fold contain {len(dataset.fold5[0])} data group.")
    # assert len(dataset.fold5) == 5
    test_labels = []
    test_preds = []
    all_test_smiles = []
    for data in tqdm(dataset.fold5, desc="Folds", total=5):
        print("############################################")
        print("Fold start.")
        train_data,test_data = data
        print(f"Shape of train_data: {train_data[0].shape}, test_data: {test_data[0].shape}")
        print(f"Shape of train_labels: {train_data[1].shape}, test_labels: {test_data[1].shape}")
        train_dataset = TensorDataset(train_data[0], train_data[1])
        # valid_dataset = TensorDataset(dataset.valid_set[0], dataset.valid_set[1])
        test_dataset = TensorDataset(test_data[0], test_data[1])
        test_smiles = test_data[2]
        dataloaders = (
            DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True,
            collate_fn=dataset.collate_normal
            ),
            # DataLoader(valid_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False,collate_fn=dataset.collate_normal),
            DataLoader(test_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False,
            collate_fn=dataset.collate_normal
            ),
            test_smiles
        )
        # print(C)
        # 2. 模型初始化
        print("Initializing model...")
        model = ModelRegression(args)

        # 3. 使用 pipeline 进行训练和预测
        print("Starting pipeline...")
        # results : {'test_labels': test_labels, 'test_preds': test_preds}
        trained_model, results = pipeline(args, dataloaders, ModelRegression)
        Chemical_data.refresh_data(results['test_preds'],test_smiles,args['split_type'])
        test_preds.extend(results['test_preds'])
        test_labels.extend(results['test_labels'])
        all_test_smiles.extend(test_smiles)
    # 4. 评估模型性能
    print("Evaluating model performance...")
    # 计算评估指标
    print("Evaluating performance on train, valid, and test datasets...")
    # print(train_labels)
    # print(train_preds)
    evaluator(all_test_smiles,test_labels, test_preds, "/root/reaction_data/yield_regression/output_tmp")

    # 保存模型和预测结果
    print("Saving trained model and prediction results...")
    # 保存训练好的模型
    # trained_model.save_checkpoint(args['model_save_path'])
    # 保存预测结果
    # with open(args['predictions_save_path'], 'wb') as f:
    #     pickle.dump(results, f)

    print("Pipeline completed successfully.")
