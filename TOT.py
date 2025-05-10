
from datasets import load_dataset
import re
import json
from tqdm import tqdm
import time

from dataset.APPSHandler import *
from prompt.template import *
from utils.utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_FILE = "/home/xjg/GraduationDesign/results/Qwen_experiments_tot.json"

class Node:
    def __init__(self, state, score=0.0):
        self.state = state  # 当前问题描述(还可能包含代码生成思路)
        self.score = score

class ToT:
    def __init__(self, tokenizer, model, k=2, b=1, T=2, public_tests=[], private_tests=[]):
        self.tokenizer = tokenizer
        self.model = model
        self.k = k # 设置每个节点的扩展子节点数目为2
        self.b = b # 这里的TOT采用BFS搜索策略，设置搜索宽度为1，即是说每一步都仅保存一个候选项
        self.T = T # 设置搜索总步数为2
        self.public_tests = public_tests
        self.private_tests = private_tests
    
    # 返回[thought-1...thought-k]格式的thought数组列表
    def thought_generator(self, node):
        if not node["thought"]: # 这里说明当前扩展的node是root,此时仅有problem
            inputs = self.tokenizer(tot_thought_generation_no_thought.format(node["problem"],self.k), return_tensors="pt")
            # print(tot_thought_generation_no_thought.format(node["problem"],self.k))
        else:
            inputs = tokenizer(tot_thought_generation_with_thought.format(self.k,node["problem"],node["thought"]), return_tensors="pt")
            # print(tot_thought_generation_with_thought.format(self.k,node["problem"],node["thought"]))

        outputs = self.model.generate(input_ids=inputs.input_ids.cuda(), 
                    attention_mask=inputs.attention_mask.cuda(),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=100000, 
                    num_return_sequences=1)
        new_thoughts = [tokenizer.decode(output[inputs.input_ids.size(1):].cpu(),
                         skip_special_tokens=True).strip() for output in outputs]
        # print(new_thoughts[0])
    
        json_str = extract_json(new_thoughts[0])
        # print(manual_json_parse(json_str))
        return manual_json_parse(json_str)


    def thought_evaluator(self, node):
        # print(thought_evaluation.format(node["problem"],node["thought"]))
        inputs = tokenizer(thought_evaluation.format(node["problem"],node["thought"]),return_tensors="pt")
        outputs = model.generate(input_ids=inputs.input_ids.cuda(), 
                            attention_mask=inputs.attention_mask.cuda(),
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=100000,
                            num_return_sequences=1) # 设置输出回答数目为3条
        response = [tokenizer.decode(output[inputs.input_ids.size(1):].cpu(),
                        skip_special_tokens=True).strip() for output in outputs]
        # print(response[0])
        # 补丁： Qwen生成的response格式还可能为xxxx. The correctness score is x.
        match = re.search(r"evaluation:\s*(-?\d+(?:\.\d+)?)| The correctness score is\s*(-?\d+(?:\.\d+)?)", response[0])
        if match:
            return float(match.group(1) if match.group(1) else match.group(2))
        else:
            raise ValueError("Failure to capture valid evaluation scores")
    
    def TOT_Thought_BFS(self, node):
        S = []
        S.append(node) # 初始节点node仅包含问题描述
        for _ in range(self.T):
            S_Temp = []
            for s in S:
                Z = self.thought_generator(s)
                for z in Z:
                    new_node = Node({"problem": s.state["problem"], "thought": z["Thought"]}, score)
                    score = self.thought_evaluator(new_node)
                    new_node.score = score
                    S_Temp.append(new_node)
            S_Temp.sort(key=lambda node: node.score, reverse=True)
            S = S_Temp[:self.b]
        # 最终按照搜索过程中得到的得分最高的thought，生成又一新thought，并将其作为最终结果返回
        Z = self.thought_generator(S[0])
        return Z[0]


def main():
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("/home/xjg/checkpoints/Qwen2.5-32B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/xjg/checkpoints/Qwen2.5-32B", device_map="auto", trust_remote_code=True)
    # 初始化TOT
    tot = TOT(tokenizer, model)
    test_data = load_proceed(100) # 各个难度的题目抽100道
    first_write = False

    for category, dataset in test_data.items():
        print(f"📂 处理类别: {category}，共 {len(dataset)} 道题目")

        # 使用 tqdm 显示进度条
        for row in tqdm(dataset, desc=f"Processing {category}"):
            if category != "introductory" or row["id"] > 4029:
                continue
            # res = run_tests(code, row["private_tests"])
            # print("here")
            # print(res)
            # # 记录结果
            # result_entry = {
            #     "category": category,
            #     "question_id": row['id'],
            #     "time_taken": "Timeout",
            #     "pass_rate": res["pass_rate"],
            #     "thought": "",
            #     "best_code": code  # 也可以存储代码
            # }
            # save_result_entry(result_entry, is_first=first_write)
            # exit()
            # if category == "introductory" and row["id"] > 4029:
            #     continue
            # if category == "interview" and row["id"] > 29:
            #     continue
            # if category == "competition" and row["id"] > 3029:
            #     continue
            question_id = row["id"]
            tot.public_tests = row["public_tests"]
            tot.private_tests = row["private_tests"]

            root = Node({"problem": row["question"], "thought": ""})

            # 按照BFS对TOT进行搜索
            best_thought = tot.TOT_Thought_BFS(root)
            _ , best_code = generate_code(tokenizer, model, row["question"], best_thought)

            # 评估私有测试集
            private_result = run_tests(best_code, row["private_tests"])

            # 记录结果
            result_entry = {
                "category": category,
                "question_id": question_id,
                "pass_rate": private_result["pass_rate"],
                "thought": best_thought,
                "best_code": best_code
            }
            save_result_entry(result_entry, RESULTS_FILE, is_first=first_write)
            first_write = False

            print(f"✅ {category} - {question_id} 处理完成，私有测试通过率: {private_result['pass_rate']:.2f}")
    finalize_results(RESULTS_FILE)
    print("🎉 所有任务处理完成！结果已保存至", RESULTS_FILE)
if __name__ == '__main__':
    main()
