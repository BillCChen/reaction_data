import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchmetrics import PearsonCorrCoef, R2Score

def pipeline(args, dataloaders, model_class):
    """
    Pipeline 函数，依次进行数据加载、数据处理、模型拟合和预测值获取
    :param args: 配置字典，包含所有必要的参数
    :param dataloaders: 包含训练集、验证集和测试集的 dataloaders
    :param model_class: 用于模型创建的类
    :return: 训练好的模型和预测结果
    """
    # 数据加载和处理
    print("Loading and preparing data...")
    # 假设 dataloaders 已经包含了 train_loader, valid_loader 和 test_loader
    train_loader, valid_loader, test_loader = dataloaders

    # 模型创建
    print("Initializing model...")
    model = model_class(args)

    # 训练模型并进行预测
    print("Training and predicting...")
    trained_model = train_func(model, dataloaders, args)

    # 获取预测值
    print("Fetching predictions...")
    model.eval()  # 切换到评估模式
    train_preds = predict(trained_model, train_loader)
    valid_preds = predict(trained_model, valid_loader)
    test_preds = predict(trained_model, test_loader)

    # 保存或返回预测结果
    results = {
        'train_preds': train_preds,
        'valid_preds': valid_preds,
        'test_preds': test_preds,
        'train_labels': [label for _, label in train_loader.dataset],
        'valid_labels': [label for _, label in valid_loader.dataset],
        'test_labels': [label for _, label in test_loader.dataset]
    }

    return trained_model, results

def evaluater(labels, preds, train_idx, valid_idx, test_idx):
    """
    计算并输出 MAE, MSE, Pearson-R 和 R2（拟合优度）评分
    :param labels: 真实标签
    :param preds: 模型预测值
    :param train_idx: 训练集索引
    :param valid_idx: 验证集索引
    :param test_idx: 测试集索引
    """
    # 计算 MAE 和 MSE
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)

    # 计算 Pearson-R 和 R2
    pearson_r = PearsonCorrCoef()(torch.tensor(preds), torch.tensor(labels))
    r2 = R2Score()(torch.tensor(preds), torch.tensor(labels))

    # 输出评估结果
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Pearson-R: {pearson_r:.4f}")
    print(f"R2: {r2:.4f}")

    return {
        'MAE': mae,
        'MSE': mse,
        'Pearson-R': pearson_r,
        'R2': r2
    }

# 预测函数
def predict(model, dataloader):
    """
    在指定的 dataloader 上进行预测。
    :param model: 已训练的模型
    :param dataloader: 用于预测的 dataloader
    :return: 预测值列表
    """
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 不需要计算梯度
        for batch in dataloader:
            x, y = batch['fingerprint'], batch['labels']
            preds = model(x)
            all_preds.append(preds.squeeze().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)
