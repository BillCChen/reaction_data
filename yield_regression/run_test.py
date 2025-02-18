import torch
from torch.utils.data import DataLoader
import pickle
import os
from model_regression import ModelRegression
from data_arrangement import DataArrangement
from chemical_reaction_data_loader import ChemicalReactionDataLoader

if __name__ == "__main__":
    # 设置一些基本的参数
    args = {
        'dataset_path': 'path_to_dataset.pkl',  # 数据集路径
        'encoding_type': 'DRFP',  # 编码类型
        'encoding_params': {'param1': 10, 'param2': 5},  # 编码参数
        'force_reencoding': False,  # 是否强制重新编码
        'split_type': 'random',  # 数据分割类型 ('random' 或 'OOD')
        'split_args': [0.7, 0.2, 0.1],  # 随机分割的比例 [训练集，验证集，测试集]
        'batch_size': 32,  # 批量大小
        'num_workers': 0,  # 工作线程数
        'max_epochs': 100,  # 最大训练轮数
        'lr': 0.001,  # 学习率
        'model_save_path': 'path_to_save_model.ckpt',  # 模型保存路径
        'predictions_save_path': 'predictions.pkl',  # 预测结果保存路径
        'gpus': None,  # GPU 设置
        'early_stopping': 'valid_loss',  # 提前停止的监控指标
    }

    # 1. 数据加载和预处理
    print("Loading dataset and preparing dataloaders...")
    loader = ChemicalReactionDataLoader(args['dataset_path'], args['encoding_type'], args['encoding_params'], args['force_reencoding'])
    fingerprints, labels = loader.get_data(data_type="for_train")
    
    # 使用 DataArrangement 处理数据
    data_arrangement = DataArrangement(args, split_type=args['split_type'], split_args=args['split_args'])
    dataloaders = (
        DataLoader(data_arrangement.train_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True),
        DataLoader(data_arrangement.valid_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False),
        DataLoader(data_arrangement.test_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False),
    )
    
    # 2. 模型初始化
    print("Initializing model...")
    model = ModelRegression(args)

    # 3. 使用 pipeline 进行训练和预测
    print("Starting pipeline...")
    trained_model, results = pipeline(args, dataloaders, ModelRegression)

    # 4. 评估模型性能
    print("Evaluating model performance...")
    train_labels, valid_labels, test_labels = results['train_labels'], results['valid_labels'], results['test_labels']
    train_preds, valid_preds, test_preds = results['train_preds'], results['valid_preds'], results['test_preds']

    # 计算评估指标
    print("Evaluating performance on train, valid, and test datasets...")
    evaluater(train_labels, train_preds, None, None, None)
    evaluater(valid_labels, valid_preds, None, None, None)
    evaluater(test_labels, test_preds, None, None, None)

    # 保存模型和预测结果
    print("Saving trained model and prediction results...")
    # 保存训练好的模型
    trained_model.save_checkpoint(args['model_save_path'])
    # 保存预测结果
    with open(args['predictions_save_path'], 'wb') as f:
        pickle.dump(results, f)

    print("Pipeline completed successfully.")
