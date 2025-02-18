import torch
import pytorch_lightning as pl
import pickle

def train_func(model, dataloaders, args):
    """
    训练模型并记录结果。
    :param model: 要训练的模型
    :param dataloaders: 包含训练集、验证集和测试集的 dataloader
    :param args: 配置字典，包括保存路径和其他参数
    :return: 最终的预测结果和标签
    """
    # 定义训练相关的变量
    train_loader, valid_loader, test_loader = dataloaders
    model_path = args['model_save_path']
    max_epochs = args.get('max_epochs', 100)
    
    # 定义训练器
    trainer = pl.Trainer(max_epochs=max_epochs, 
                         gpus=args.get('gpus', None), 
                         logger=None,
                         early_stop_callback=False)

    # 训练模型
    print("Starting training process...")
    trainer.fit(model, train_loader, valid_loader)

    # 训练后保存模型
    print(f"Saving trained model to {model_path}...")
    trainer.save_checkpoint(model_path)
    
    # 训练完模型后，我们需要在测试集上进行预测并记录
    print("Making predictions on test set...")

    # 使用 train 数据训练得到的模型进行预测
    model.eval()  # 切换到评估模式
    train_preds = predict(model, train_loader)
    valid_preds = predict(model, valid_loader)
    test_preds = predict(model, test_loader)
    
    # 使用 train+valid 数据重新训练模型
    print("Retraining with train+valid data...")
    trainer.fit(model, torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([train_loader.dataset, valid_loader.dataset]),
        batch_size=args.get('batch_size', 32),
        num_workers=args.get('num_workers', 0)
    ))

    # 重新拟合后再次进行预测
    print("Making predictions on test set after retraining...")
    retrained_preds = predict(model, test_loader)

    # 保存最终的预测结果和标签
    print("Saving predictions and labels to pkl file...")
    results = {
        'labels': [label for _, label in test_loader.dataset],
        'train_preds': train_preds,
        'valid_preds': valid_preds,
        'test_preds_before_retrain': test_preds,
        'test_preds_after_retrain': retrained_preds
    }

    # 保存预测结果
    with open(args['predictions_save_path'], 'wb') as f:
        pickle.dump(results, f)

    print("Training and prediction process completed.")
    return model

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
