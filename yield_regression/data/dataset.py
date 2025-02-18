import os
import pickle
import numpy as np

class ChemicalReactionDataset:
    def __init__(self, dataset_path: str, encoding_type: str, encoding_param: dict, force_reencoding: bool):
        """
        初始化类，读取数据并检查有效性，进行编码处理。
        :param dataset_path: 数据集路径 (字符串)，指向一个 pkl 文件
        :param encoding_type: 编码类型 ('DRFP', 'unirxn', 'renfp')
        :param encoding_param: 编码所需的参数（字典）
        :param force_reencoding: 是否强制重新编码 (布尔值)
        """
        # 参数校验
        self._validate_parameters(dataset_path, encoding_type, encoding_param, force_reencoding)

        self.dataset_path = dataset_path
        self.encoding_type = encoding_type
        self.encoding_param = encoding_param
        self.force_reencoding = force_reencoding
        
        # 读取数据
        self.data = self._load_data()

        # 执行编码
        self._encoding()

    def _validate_parameters(self, dataset_path, encoding_type, encoding_param, force_reencoding):
        """验证参数的有效性"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")
        
        valid_encoding_types = ['DRFP', 'unirxn', 'renfp']
        if encoding_type not in valid_encoding_types:
            raise ValueError(f"Invalid encoding type. Valid types are {valid_encoding_types}.")
        
        if not isinstance(encoding_param, dict):
            raise TypeError("Encoding parameters should be a dictionary.")
        
        if not isinstance(force_reencoding, bool):
            raise TypeError("force_reencoding should be a boolean value.")
        
    def _load_data(self):
        """加载数据集"""
        print("Loading data from pkl file...")
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Data in pkl file should be a list.")
        
        return data

    def get_data(self, mode="for_train"):
        """获取数据的方法，根据模式返回不同的数据"""
        if mode == "for_train":
            print("Extracting data for training...")
            encoding_list = []
            label_list = []
            for item in self.data:
                encoding_list.append(item.get('encoding', {}).get(self.encoding_type, {}).get(str(self.encoding_param), []))
                label_list.append(item.get('label', []))
            return encoding_list, label_list
        elif mode == "for_analyze":
            print("Returning all data for analysis...")
            return self.data
        else:
            raise ValueError("Invalid mode. Choose 'for_train' or 'for_analyze'.")

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
        for i, item in enumerate(self.data):
            if 'encoding' not in item or not item['encoding'].get(self.encoding_type, {}).get(str(self.encoding_param)):
                print(f"Encoding reaction {i}...")
                tensor = self._get_encoding_tensor(item['SMILES'])
                if 'encoding' not in item:
                    item['encoding'] = {}
                if self.encoding_type not in item['encoding']:
                    item['encoding'][self.encoding_type] = {}
                item['encoding'][self.encoding_type][str(self.encoding_param)] = tensor
            else:
                print(f"Skipping encoding for reaction {i}, already encoded.")
        self._save_data()
    def _encoding_in_docking(self):
            # 使用 Docker 进行编码
            print("Encoding in Docker...")
            docker_input_file = "docker_encoding_input.pkl"
            
            # 检查是否已经有输入文件
            if not os.path.exists(docker_input_file) or self.force_reencoding:
                with open(docker_input_file, 'wb') as f:
                    pickle.dump([entry["smiles"] for entry in self.data], f)
                print(f"SMILES data written to {docker_input_file}")
            
            # 调用 Docker 进行编码
            from docker import Run_encoding  # 假设该函数位于 docker.py 文件中
            output_file = Run_encoding(self.encoding_type, docker_input_file, self.encoding_params)

            # 加载编码结果并更新数据
            with open(output_file, 'rb') as f:
                encoded_data = pickle.load(f)

            for i, entry in enumerate(self.data):
                entry["encoding"] = encoded_data[i]

            # 更新 pkl 文件
            with open(self.dataset_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Encoding in Docker completed and results saved to {self.dataset_path}")

    def _get_encoding_tensor(self, smiles: str):
        """根据 SMILES 编码化学反应为一维张量"""
        # 根据编码类型选择相应的编码函数
        encoding_func = self._get_encoding_func()
        return encoding_func(self.encoding_param, smiles)

    def _get_encoding_func(self):
        """根据编码类型导入对应的编码函数"""
        if self.encoding_type == 'DRFP':
            from encoding_module import encoding_by_DRFP
            return encoding_by_DRFP
        elif self.encoding_type == 'unirxn':
            from encoding_module import encoding_by_unirxn
            return encoding_by_unirxn
        elif self.encoding_type == 'renfp':
            from encoding_module import encoding_by_renfp
            return encoding_by_renfp
        else:
            raise ValueError("Unknown encoding type.")

    def _save_data(self):
        """保存更新后的数据"""
        print("Saving encoded data to pkl file...")
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.data, f)

# 示例使用
# loader = ChemicalReactionDataLoader("data.pkl", "DRFP", {"param1": 10}, force_reencoding=True)
# loader.analyze_data()
# encoding_data = loader.get_data("for_train")
