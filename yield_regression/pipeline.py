import torch
import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchmetrics import PearsonCorrCoef, R2Score
import torch.nn.functional as F
def pipeline(args, dataloaders, model_class,jump_train=False):
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
    train_loader, test_loader , smiles = dataloaders

    # 模型创建
    print("Initializing model...")
    model = model_class(args)

    # 训练模型并进行预测
    print("Training and predicting...")
    trained_model = train_func(model, dataloaders, args)

    # 获取预测值
    print("Fetching predictions...")
    model.eval()  # 切换到评估模式
    # train_preds,train_labels = predict(trained_model, train_loader)
    # valid_preds,valid_labels = predict(trained_model, valid_loader)
    test_preds,test_labels = predict(trained_model, test_loader)

    # 保存或返回预测结果
    results = {
        # 'train_preds': train_preds,
        # 'valid_preds': valid_preds,
        'test_preds': test_preds,
        # 'train_labels': train_labels,
        # 'valid_labels': valid_labels,
        'test_labels': test_labels
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
    # 计算 MAE 和 MSE,使用 numpy
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    

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
def train_func(model, dataloaders, args):
    # 配置优化器和学习率调度器
    train_loader, test_loader ,smiles = dataloaders
    num_epochs = args['max_epochs']
    train_parts = model.return_train_parts()
    optimizer = train_parts['optimizer']
    lr_scheduler = train_parts['lr_scheduler']
    early_stopping = train_parts['early_stopping']
    valid_losses = []
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        if not args['jump_train'] == "True":
            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch,None)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                loss = model.validation_step(batch,None)
                valid_loss += loss.item()
        valid_loss /= len(test_loader)
        valid_losses.append(valid_loss)

        # 更新学习率
        lr_scheduler.step()

        # 打印训练和验证损失
        model.on_epoch_end(epoch)

        # 提前停止
        # if early_stopping == 'valid_loss':
        #     if epoch > 0 and valid_loss > min(valid_losses):
        #         print("Early stopping activated.")
        #         break

    # 测试
    model.eval()
    test_loss = 0
    r2_score = R2Score()
    pearson_r = PearsonCorrCoef()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['fingerprint'], batch['labels'] 
            y_hat = model(x)
            loss = F.mse_loss(y_hat.squeeze(), y)
            test_loss += loss.item()
            r2_score.update(y_hat.squeeze(), y)
            pearson_r.update(y_hat.squeeze(), y)

    test_loss /= len(test_loader)
    final_r2 = r2_score.compute().item()
    final_pearson_r = pearson_r.compute().item()

    print(f"Test Loss: {test_loss:.4f}, R2: {final_r2:.4f}, Pearson R: {final_pearson_r:.4f}")

    return model