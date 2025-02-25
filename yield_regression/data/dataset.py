import torch
import os
import pickle
import numpy as np

class ChemicalReactionDataset:
    def __init__(self, args,dataset_path: str, encoding_type: str, force_reencoding: bool):
        """
        初始化类，读取数据并检查有效性，进行编码处理。
        :param dataset_path: 数据集路径 (字符串)，指向一个 pkl 文件
        :param encoding_type: 编码类型 ('DRFP', 'unirxn', 'renfp')
        :param encoding_param: 编码所需的参数（字典）
        :param force_reencoding: 是否强制重新编码 (布尔值)
        """
        # 参数校验
        self._validate_parameters(dataset_path, encoding_type, force_reencoding)
        self.args = args
        self.dataset_path = dataset_path
        self.encoding_type = encoding_type
        self.force_reencoding = force_reencoding
        
        # 读取数据
        self._load_data()

        # 执行编码
        self._encoding()

    def _validate_parameters(self, dataset_path, encoding_type, force_reencoding):
        """验证参数的有效性"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")
        
        valid_encoding_types = ['DRFP', 'unirxn', 'renfp']
        if encoding_type not in valid_encoding_types:
            raise ValueError(f"Invalid encoding type. Valid types are {valid_encoding_types}.")
        if not isinstance(force_reencoding, bool):
            raise TypeError("force_reencoding should be a boolean value.")
    def check_keys(self,data):
        """检查数据是否有必要的键,报错说明数据集源文件有问题"""
        if not isinstance(data, dict):
            raise ValueError("Data in pkl file should be a dict.Data should have {reaction1:{{}},reaction2:{{}}} format.")
        for key,item in data.items():
            if 'smiles' not in item.keys():
                data[key]['smiles'] = key
            if 'label' not in item.keys():
                raise ValueError("Data should have 'label' key.")
            if 'encoding'  not in item.keys():
                data[key]['encoding'] = {
                    'info': "encoding format as {DRFP:tensor}} or {function:tensor}}",
                }
            if 'predict' not in item.keys():
                data[key]['predict'] = {
                    'info': "predict format as {DRFP:value}} or {function:value}}",
                }
        return data
    def _load_data(self):
        """加载数据集"""
        print("Loading data from pkl file...")
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        data = self.check_keys(data)
        self.data = data

    def get_data(self, *args, **kwargs):
        """获取数据的方法，根据模式返回不同的数据"""
        data_dict = {}
        for arg in args:
            data_dict[arg] = []
        print("Extracting data for training...")
        for key,item in self.data.items():
            for arg in args:
                if arg == 'encoding':
                    data_dict[arg].append(item[arg][self.encoding_type])
                else:
                    data_dict[arg].append(item[arg])
        return data_dict

    def analyze_data(self):
        """分析数据集的标签分布"""
        print("Analyzing data...")
        labels = [item.get('label', None) for item in self.data]
        labels = [label for label in labels if label is not None]
        
        # 标签直方图
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Label histogram: {dict(zip(unique_labels, counts))}")
        
        # 标签的统计分析
        mean = np.mean(labels)
        q1 = np.percentile(labels, 25)
        q3 = np.percentile(labels, 75)
        print(f"Mean: {mean}")
        print(f"1st Quartile: {q1}")
        print(f"3rd Quartile: {q3}")

    def _encoding(self):
        """执行编码"""
        print(f"Encoding data using method '{self.encoding_type}'...")
        if not self.force_reencoding:
            print("Data already encoded. Skipping encoding...")
            return

        for i,(key, item) in enumerate(self.data.items()):
            if i % 33 == 0:
                print(f"Encoding reaction {i}...")
            tensor = self._get_encoding_tensor(item['smiles'])
            
            self.data[key]['encoding'][self.encoding_type] = torch.tensor(tensor[0].reshape(-1),dtype=torch.float32)

        self._save_data()

    def _get_encoding_tensor(self, smiles: str):
        """根据 SMILES 编码化学反应为一维张量"""
        # 根据编码类型选择相应的编码函数
        result = self._get_encoding_func(smiles)
        return result
    def link_unirxn(self,encoding_type):
        if encoding_type == 'unirxn':
            "导入 unirxn 编码得到的 smiles 和编码的 dict"
            with open('unirxn.pkl','rb') as f:
                encoding_book = pickle.load(f)
            self.encoding_book = encoding_book
        elif encoding_type == 'rxnfp':
            "导入 rxnfp 编码得到的 smiles 和编码的 dict"
            with open('rxnfp.pkl','rb') as f:
                encoding_book = pickle.load(f)
            self.encoding_book = encoding_book
        else:
            raise ValueError("Unknown encoding type.")
    def _get_encoding_func(self,smiles):
        """根据编码类型导入对应的编码函数"""
        # self.encoding_param, self.encoding_type
        if self.encoding_type == 'DRFP':
            from drfp import DrfpEncoder
            return DrfpEncoder.encode(smiles)
        elif self.encoding_type == 'unirxn':
            self.link_unirxn("unirxn")
            return self.encoding_book(smiles)
        elif self.encoding_type == 'rxnfp':
            self.link_unirxn("rxnfp")
            return self.encoding_book(smiles)
        else:
            raise ValueError("Unknown encoding type.")
    def refresh_data(self,results,smiles,split_type):
        """更新数据集的预测值"""
        for key,item in self.data.items():
            if self.encoding_type not in item['predict'].keys():
                self.data[key]['predict'][self.encoding_type] = {
                    'random': None,
                    'OOD': None
                }
        for i in range(len(results)):
            self.data[smiles[i]]['predict'][self.encoding_type][split_type] = results[i]
        self._save_data()
    def _save_data(self):
        """保存更新后的数据"""
        print("Saving encoded data to pkl file...")
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.data, f)

# 示例使用
# loader = ChemicalReactionDataLoader("data.pkl", "DRFP", {"param1": 10}, force_reencoding=True)
# loader.analyze_data()
# encoding_data = loader.get_data("for_train")
