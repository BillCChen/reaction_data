import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import numpy as np
import random

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import DataStructs
from rdkit.Chem import AllChem
# 1. 数据加载模块（添加数据增强）
class MoleculeDataset(Dataset):
    atom_types = {'C': 0, 'O': 1, 'F': 2, 'N': 3, 'S': 4, 'Cl': 5, 'Br': 6}
    
    def __init__(self, smiles_list, augment=True):
        self.smiles_list = smiles_list
        self.augment = augment
        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=90),
        # ])

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.RemoveHs(mol)
        # 除去氢原子
        mol = Chem.RemoveHs(mol)
        num_atoms = mol.GetNumAtoms()

        # 生成图像
        img = Draw.MolToImage(mol, size=(500, 500))
        img = np.array(img).transpose(2, 0, 1) 
        
        img = torch.FloatTensor(img) / 255.0

        # 数据增强
        if self.augment:
            # img = self.transform(img)
            pass

        # 生成标签矩阵
        label = torch.full((100, 100), -1.0)
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for i in range(num_atoms):
            label[i, i] = self.atom_types[atoms[i]]
            for j in range(i + 1, num_atoms):
                if mol.GetBondBetweenAtoms(i, j):
                    label[i, j] = 1.0
                    label[j, i] = 1.0
                else:
                    label[i, j] = 0.0
                    label[j, i] = 0.0

        # 生成 mask
        mask = torch.zeros(100, 100)
        mask[:num_atoms, :num_atoms] = 1.0

        return img, label, mask, smiles

# 2. 模型架构（保持不变）
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Decoder
        self.upconv = nn.ConvTranspose2d(256, 128, 4, stride=4)
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.final = nn.Conv2d(128, 1, 1)
        
        # Custom activation
        self.register_buffer('diag_mask', torch.eye(100).bool().unsqueeze(0))

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # (B,64,100,100)
        x2 = self.pool(x1) # (B,64,50,50)
        x2 = self.enc2(x2) # (B,128,50,50)
        x3 = self.pool(x2) # (B,128,25,25)
        x3 = self.enc3(x3) # (B,256,25,25)
        
        # Decoder
        u = self.upconv(x3)  # (B,128,100,100)
        u = torch.cat([u, x1], dim=1)  # (B,192,100,100)
        u = self.dec1(u)     # (B,128,100,100)
        out = self.final(u)  # (B,1,100,100)
        
        # Custom activation
        out = torch.where(self.diag_mask, out, torch.sigmoid(out))
        return out.squeeze(1)  # (B,100,100)

# 3. 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_loss = nn.MSELoss()  # 对角线分类损失
        self.bce_loss = nn.BCELoss()  # 非对角线二元交叉熵损失

    def forward(self, preds, labels, masks):
        diag_mask = torch.eye(100, dtype=bool).to(preds.device)
        # print(f"diag_mask shape: {diag_mask.shape}")
        # 把 diag_mask 扩展到 batch 维度
        diag_mask = diag_mask.unsqueeze(0).expand(preds.shape[0], -1, -1)
        # print(f"preds shape: {preds.shape}")
        # print(f"labels shape: {labels.shape}")
        # 对角线分类损失
        diag_preds = preds[diag_mask].view(-1, 100)  # 7 类原子
        # 对diag_preds进行取整并转换为 long tensor
        # diag_preds = torch.floor(diag_preds).long()
        # print(diag_preds[0,:100])
        # print(f"diag_preds shape: {diag_preds.shape}")
        diag_labels = labels[diag_mask].view(-1, 100)
        # print(diag_labels[0,:100])
        # print(f"diag_labels shape: {diag_labels.shape}")
        class_loss = self.class_loss(diag_preds, diag_labels)
        
        # 非对角线二元交叉熵损失
        # off_diag_mask = masks & ~diag_mask
        off_diag_mask = masks.bool() & ~diag_mask
        # print(off_diag_mask[0,:10,:10])
        # print(f"off_diag_mask shape: {off_diag_mask.shape}")
        off_diag_preds = preds[off_diag_mask]
        # print(off_diag_preds[:100])
        # print(f"off_diag_preds shape: {off_diag_preds.shape}")
        off_diag_labels = labels[off_diag_mask]
        
        # print(off_diag_labels[:100])
        # print(f"off_diag_labels shape: {off_diag_labels.shape}")
        # print(C)
        
        off_diag_preds = torch.clamp(off_diag_preds, 1e-7, 1 - 1e-7)
        off_diag_labels = torch.clamp(off_diag_labels, 0.0, 1.0)
        bce_loss = self.bce_loss(off_diag_preds, off_diag_labels)
        print(f"class_loss: {class_loss}, bce_loss: {bce_loss}")
        return class_loss + bce_loss

# 4. 训练流程（支持多 GPU 并行训练）
def train():
    # 参数设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    lr = 3e-4
    epochs = 50
    
    # 数据加载
    import pickle
    with open("/root/tmp_smi.pkl", "rb") as f:
        smiles_list = pickle.load(f)
    dataset = MoleculeDataset(smiles_list, augment=True)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    
    # 模型初始化（多 GPU 并行）
    model = UNet().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # 优化器和学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 损失函数
    criterion = CustomLoss()
    
    # 训练循环
    from tqdm import tqdm
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, labels, masks, _ in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100,total=len(loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, labels, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 学习率调度
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# 5. 测试指标（保持不变）
def evaluate(model, loader, device):
    model.eval()
    mse_total = 0.0
    similarity_total = 0.0
    count = 0
    
    with torch.no_grad():
        for imgs, labels, masks, smiles_list in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            
            # MSE计算
            processed_preds = post_process(preds)
            mse = ((processed_preds - labels)**2 * masks).sum() / masks.sum()
            mse_total += mse.item()
            
            # 相似度计算
            for i in range(len(smiles_list)):
                pred_smiles = matrix_to_smiles(processed_preds[i])
                true_smiles = smiles_list[i]
                similarity = compute_similarity(pred_smiles, true_smiles)
                similarity_total += similarity
                count += 1
                
    return mse_total / len(loader), similarity_total / count

def post_process(matrix):
    matrix = matrix.clone()
    diag = torch.eye(100, dtype=bool)
    matrix[diag] = torch.floor(matrix[diag])
    matrix[matrix < 0] = -1
    matrix[~diag] = (torch.sigmoid(matrix[~diag]) >= 0.9).float()
    return matrix


def matrix_to_smiles(matrix):
    """
    将 100x100 的矩阵转换为 SMILES 字符串。
    :param matrix: 100x100 的矩阵，对角线为原子类型，非对角线为键信息
    :return: SMILES 字符串
    """
    # 原子类型映射
    atom_types = {0: 'C', 1: 'O', 2: 'F', 3: 'N', 4: 'S', 5: 'Cl', 6: 'Br'}
    # 对角线为原子类型，非对角线为键信息
    # 对对角线取整
    # 构建对角线掩码
    diag_mask = torch.eye(100, dtype=bool)
    matrix[diag_mask] = torch.floor(matrix[diag_mask])
    # 对角线以外的区域大于0.5的设置为1，小于0.5的设置为0
    matrix[~diag_mask] = ((matrix[~diag_mask]) >= 0.5).float()
    # 提取有效区域
    num_atoms = 0
    for i in range(100):
        if matrix[i, i] >= 0:
            num_atoms += 1
        else:
            break
    
    if num_atoms == 0:
        return None  # 无效分子
    
    # 创建分子对象
    mol = Chem.RWMol()
    
    # 添加原子
    for i in range(num_atoms):
        atom_type = int(matrix[i, i])
        if atom_type not in atom_types:
            return None  # 无效原子类型
        atom_symbol = atom_types[atom_type]
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)
    
    # 添加键
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if matrix[i, j] == 1:  # 存在键
                mol.AddBond(i, j, rdchem.BondType.SINGLE)
    
    # 转换为 SMILES
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return None  # 转换失败


def compute_similarity(smiles1, smiles2):
    """
    计算两个 SMILES 字符串的 ECFP4 相似度（Tanimoto 系数）。
    :param smiles1: 第一个 SMILES 字符串
    :param smiles2: 第二个 SMILES 字符串
    :return: ECFP4 相似度（0 到 1 之间）
    """
    # 将 SMILES 转换为分子对象
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
    except:
        return 0.0
    
    if mol1 is None or mol2 is None:
        return 0.0  # 无效分子
    
    # 生成 ECFP4 指纹
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    
    # 计算 Tanimoto 相似度
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def main():
    # 1. 参数设置
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    lr = 3e-3
    epochs = 50
    num_workers = 4  # 数据加载的线程数

    # 2. 加载数据集
    import pickle
    with open("/root/tmp_smi.pkl", "rb") as f:
        smiles_list = pickle.load(f)
    train_dataset = MoleculeDataset(smiles_list, augment=True)
    val_dataset = MoleculeDataset(smiles_list, augment=False)  # 验证集不需要数据增强

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. 初始化模型、优化器和损失函数
    model = UNet().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = CustomLoss()

    # 4. 训练循环
    best_val_loss = float('inf')
    from tqdm import tqdm
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for imgs, labels, masks, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100,total=len(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            similarity = 0.0
            cnt = 0
            for imgs, labels, masks, smiles in tqdm(val_loader, desc='Validation', ncols=100,total=len(val_loader)):
                imgs = imgs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                predict_smiles = matrix_to_smiles(preds[0])
                similarity += compute_similarity(predict_smiles, smiles[0])
                loss = criterion(preds, labels, masks)
                val_loss += loss.item()
                print(f"True SMILES: {smiles[0]}",end="  ")
                print(f"Predicted SMILES: {predict_smiles}")
                cnt += 1
                if cnt == 100:
                    break
            similarity /= 100
        avg_val_loss = val_loss / 100

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 打印训练和验证损失
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"ECFP4 Similarity: {similarity:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model!")

    # 5. 测试阶段
    print("Testing the best model...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_mse, test_similarity = evaluate(model, val_loader)
    print(f"Test MSE: {test_mse:.4f}, Test ECFP4 Similarity: {test_similarity:.4f}")

if __name__ == "__main__":
    main()