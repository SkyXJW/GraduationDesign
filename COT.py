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

RESULTS_FILE = "/home/xjg/GraduationDesign/results/Qwen_experiments_cot.json"

def main():
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("/home/xjg/checkpoints/Qwen2.5-32B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/xjg/checkpoints/Qwen2.5-32B", device_map="auto", trust_remote_code=True)
    
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

            # print(cot_prompt.format(problem))
            inputs = tokenizer(cot_prompt.format(problem),return_tensors="pt")
            outputs = model.generate(input_ids=inputs.input_ids.cuda(), 
                        attention_mask=inputs.attention_mask.cuda(),
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=100000,
                        num_return_sequences=1) # 设置输出回答数目为3条
            response = [tokenizer.decode(output[inputs.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for output in outputs]
            # print(response[0])

            # 提取code
            match = re.search(r"<code>(.*?)</code>|```python\s+([\s\S]*?)```", response[0], re.DOTALL)
            if match:
                code = match.group(1) if match.group(1) else match.group(2)
            else:
                # 实际实验过程中发现，LLM给出的代码也会直接裸露在外，不被包含在任何标签中
                code = response[0].strip()
                # raise ValueError("Failure to capture valid code")

            # 提取thought
            match = re.search(r"<idea>(.*?)</idea>|###\s*Approach\s+(.*?)###\s*Solution Code", response[0], re.DOTALL)
            if match:
                thought = match.group(1) if match.group(1) else match.group(2).strip()
            else:
                raise ValueError("Failure to capture valid thought")

            # 评估私有测试集
            private_result = run_tests(code, row["private_tests"])

            # 记录结果
            result_entry = {
                "category": category,
                "question_id": question_id,
                "pass_rate": private_result["pass_rate"],
                "thought": thought,
                "code": code
            }
            save_result_entry(result_entry, RESULTS_FILE, is_first=first_write)
            first_write = False

            print(f"✅ {category} - {question_id} 处理完成，私有测试通过率: {private_result['pass_rate']:.2f}")
    finalize_results(RESULTS_FILE)
    print("🎉 所有任务处理完成！结果已保存至", RESULTS_FILE)
if __name__ == '__main__':
    main()