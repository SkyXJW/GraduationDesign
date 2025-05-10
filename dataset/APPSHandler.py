from datasets import load_dataset
import json

def get_raw_data_path():
    return "/home/xjg/GraduationDesign/dataset/APPS"

def load_proceed(num):
     # 加载APPS数据集并按难度划分
    dataset = load_dataset(get_raw_data_path())
    test_data = {
        "introductory": dataset["test"].filter(lambda x: x["difficulty"] == "introductory").select(range(num)),
        "interview": dataset["test"].filter(lambda x: x["difficulty"] == "interview").select(range(num)),
        "competition": dataset["test"].filter(lambda x: x["difficulty"] == "competition").select(range(num)),
    }
    # 对每个 Dataset 进行 map 变换
    test_data = {
        key: ds.map(split_test_cases)
        for key, ds in test_data.items()
    }
    return test_data

# 分割公共测试用例和私有测试用例
def split_test_cases(item):
    """处理 Dataset 的单个 item, 添加 public_tests 和 private_tests"""
    data = json.loads(item["input_output"])
    inputs = data["inputs"]
    outputs = data["outputs"]
    
    mid = len(inputs) // 2
    item["public_tests"] = [{"input": inputs[i], "output": outputs[i]} for i in range(mid + 1)]
    item["private_tests"] = [{"input": inputs[i], "output": outputs[i]} for i in range(mid + 1, len(inputs))]

    return item  # 必须返回修改后的字典