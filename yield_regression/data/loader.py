import os
import pickle
import numpy as np
import pandas as pd

class ChemicalReactionDataLoader:
    def __init__(self, dataset_path: str, encoding_type: str, encoding_params: dict, force_reencoding: bool):
        # 1. 初始化参数检查
        self.dataset_path = dataset_path
        self.encoding_type = encoding_type
        self.encoding_params = encoding_params
        self.force_reencoding = force_reencoding

        # 2. 检查文件有效性
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        # 3. 加载数据
        self.data = self.load_data()

        # 4. 调用编码方法
        self.encoding()

    def load_data(self):
        # 读取pkl文件，返回数据
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Dataset should be a list of dictionaries")
        
        print(f"Loaded dataset with {len(data)} records.")
        return data

    def get_data(self, data_type="for_train", ratio=None):
        # 返回指定格式的数据
        if data_type == "for_train":
            encoding_data = [entry["encoding"][self.encoding_type] for entry in self.data]
            labels = [entry["label"] for entry in self.data]
            return encoding_data, labels
        elif data_type == "for_analyze":
            return self.data
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def analyze_data(self):
        # 分析数据的标签分布及相关统计
        labels = [entry["label"] for entry in self.data]
        label_series = pd.Series(labels)
        
        print("Label Distribution:")
        print(label_series.value_counts())
        
        print("\nLabel Statistics:")
        print(f"Mean: {label_series.mean()}")
        print(f"Median: {label_series.median()}")
        print(f"25th Percentile: {label_series.quantile(0.25)}")
        print(f"75th Percentile: {label_series.quantile(0.75)}")
        print(f"Standard Deviation: {label_series.std()}")

    def encoding(self):
        # 根据编码类型进行编码
        encoding_method = self._get_encoding_method()
        
        # 检查是否需要重新编码
        if self._should_reencode():
            print(f"Re-encoding dataset using {self.encoding_type}...")
            self._apply_encoding(encoding_method)
        else:
            print("Encoding already exists, skipping re-encoding.")

    def _get_encoding_method(self):
        # 根据编码类型加载对应的编码函数
        encoding_function_name = f"encoding_by_{self.encoding_type}"
        try:
            encoding_function = globals()[encoding_function_name]
        except KeyError:
            raise ValueError(f"Encoding method {encoding_function_name} not found.")
        return encoding_function

    def _should_reencode(self):
        # 检查是否需要重新编码
        for entry in self.data:
            if "encoding" not in entry or not entry["encoding"].get(self.encoding_type):
                return True
            tensor = entry["encoding"][self.encoding_type]
            if np.all(tensor == 0):
                return True
        return False

    def _apply_encoding(self, encoding_function):
        # 应用编码
        for entry in self.data:
            smiles = entry["smiles"]
            encoded_tensor = encoding_function(self.encoding_params, smiles)
            entry["encoding"] = {self.encoding_type: encoded_tensor}
        
        # 输出编码相关信息
        print(f"Encoding completed using {self.encoding_type} with parameters {self.encoding_params}")

        # 将编码结果存储到pkl文件
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Updated encoded data stored at {self.dataset_path}")

# 假设有一个编码函数
def encoding_by_DRFP(encoding_params, smiles):
    # 假设这里是某个真实的编码方法，返回一个模拟的一维tensor
    return np.random.rand(128)  # 返回一个随机的一维tensor

def encoding_by_unirxn(encoding_params, smiles):
    # 假设这里是某个真实的编码方法，返回一个模拟的一维tensor
    return np.random.rand(256)  # 返回一个随机的一维tensor

def encoding_by_renfp(encoding_params, smiles):
    # 假设这里是某个真实的编码方法，返回一个模拟的一维tensor
    return np.random.rand(512)  # 返回一个随机的一维tensor

# 使用实例
dataset_path = "chemical_reaction_data.pkl"
encoding_type = "DRFP"
encoding_params = {"param1": 10, "param2": 5}  # 示例参数
force_reencoding = False

# 实例化类
loader = ChemicalReactionDataLoader(dataset_path, encoding_type, encoding_params, force_reencoding)
