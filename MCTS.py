import math
from collections import defaultdict

from datasets import load_dataset
import json
import re
from openai import OpenAI
from tqdm import tqdm
import time

from dataset.APPSHandler import *
from utils.utils import *
from prompt.template import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_FILE = "/home/xjg/GraduationDesign/results/Qwen_ablation_no_cut.json"

class Node:
    def __init__(self, state, parent=None, policy_prob=None):
        self.state = state  # 当前问题描述、代码生成思路(被选中经过评估阶段后，还会包含生成代码以及该代码在公共测试用例集上的测试反馈)
        self.parent = parent
        self.policy_prob = 0.0 if not policy_prob else policy_prob
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.pass_rate = 0.0  # (公共)测试用例通过率

class MCTS:
    def __init__(self, tokenizer, model, c_puct=1.0, alpha=1.0, a=0.8, b=0.2, v_pass=0.7, ground_truth_solution="", public_tests=[], private_tests=[], no_solutions=False, max_reward=-float('inf')):
        self.tokenizer = tokenizer
        self.model = model
        self.c_puct = c_puct
        self.alpha = alpha
        self.a = a
        self.b = b
        self.v_pass = v_pass
        self.ground_truth_solution = ground_truth_solution
        self.public_tests = public_tests
        self.private_tests = private_tests
        self.no_solutions = no_solutions
        self.max_reward = max_reward
        self.nodes = defaultdict(lambda: Node(None))
    
    def select(self, node):
        while node.children:
            # 选择PUCB值最高的子节点
            max_pucb = -float('inf')
            selected_child = None
            for child in node.children:
                if child.visits == 0: #这里可以理解为，当前节点的visits=0时，其对应的Q(s,a)=total_reward/visits则是无穷大，所以直接选择当前节点
                    selected_child = child
                    break
                # if child.pass_rate < self.v_pass:
                #     continue
                q = child.total_reward / child.visits
                p = child.policy_prob
                u = self.c_puct * p * math.sqrt(math.log(node.visits)) / (1 + child.visits)
                pucb = q + u + self.alpha * compare_code_similarity(child.state["code"], self.ground_truth_solution)
                # pucb = q + u
                if pucb > max_pucb:
                    if selected_child is None:
                        max_pucb = pucb
                        selected_child = child
                    # # 在child的pass_rate低于给定阈值v_pass的情况下，当且仅当child.pass_rate更高时，才会被选中
                    # # 即是说，这时pucb的优先级 < pass_rate的优先级
                    # elif child.pass_rate < self.v_pass:
                    #     if child.pass_rate > selected_child.pass_rate:
                    #         max_pucb = pucb
                    #         selected_child = child
                    # # 在child的pass_rate不低于给定阈值v_pass的情况下，即使child的pass_rate可能会更低，也依然会被选中
                    # # 即是说，这时pucb的优先级 > pass_rate的优先级
                    else:
                        max_pucb = pucb
                        selected_child = child
            node = selected_child
        return node
    
    def expand(self, node, k=2):
        # 调用LLM生成k个新思路
        new_thoughts = generate_thought(self.tokenizer, self.model, node.state["problem"], node.state["thought"], node.state["code"], node.state["feedback"], k)
        # 添加新节点
        for thought in new_thoughts:
            new_state = {"problem": node.state["problem"], "thought": thought["Thought"], "code": "", "feedback": ""}
            new_node = Node(new_state, parent=node, policy_prob=thought["Reasonableness"])
            node.children.append(new_node)
        # 完成扩展后，默认将当前扩展得到的第一个子节点作为本次被选中的节点，用于后续的evaluate、backpropagate
        return node.children[0]
    
    def evaluate(self, node):
        # 生成代码并测试
        # 这里需要考虑到由于初始节点root并不包含任何thought，因此需要LLM Agent生成
        #***但经过后续的代码修正，初始节点root永远不会被选中而用于evaluate，所以上述考虑可以忽略***
        thought, code = generate_code(self.tokenizer, self.model, node.state["problem"], node.state["thought"])
        node.state["thought"] = thought
        node.state["code"] = code
        test_result = run_tests(code, self.public_tests)
        # 计算通过率和LLM自评分数
        pass_rate = test_result["pass_rate"]
        node.state["feedback"] = test_result["failed_cases"]
        if pass_rate == 1.0:
            llm_score = self_assess(self.tokenizer, self.model, node.state["problem"], code)
            reward = self.a * pass_rate + self.b * llm_score
        else:
            reward = pass_rate
        # reward = pass_rate        
        if self.no_solutions and self.max_reward <= reward:
            self.max_reward = reward
            self.ground_truth_solution = code
        return reward
    
    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

def main():
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("/home/xjg/checkpoints/Qwen2.5-32B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/xjg/checkpoints/Qwen2.5-32B", device_map="auto", trust_remote_code=True)
    # 初始化MCTS
    mcts = MCTS(tokenizer, model)
    test_data = load_proceed(100) # 各个难度的题目抽100道
    first_write = False

    code = '''
def evaluate_bracket_sequence(n, tokens):
    stack = []
    current_op = '+'
    current_num = 0
    for token in tokens:
        if token == '(':
            stack.append(current_op)
            current_op = '+'
            current_num = 0
        elif token == ')':
            if current_op == '+':
                current_num = sum(stack.pop() for _ in range(len(stack)))
            else:
                current_num = reduce(lambda x, y: x * y, stack.pop() for _ in range(len(stack)))
            current_op = stack.pop()
        else:
            current_num += int(token)
    if current_op == '+':
        return sum(stack) + current_num
    else:
        return reduce(lambda x, y: x * y, stack) * current_num
n = int(input())
tokens = input().split()
result = evaluate_bracket_sequence(n, tokens)
print(result % (10**9 + 7))
'''
    failed_cases = []
    failed_cases.append({'input': '8\n1 2\n2 3\n3 4\n4 5\n4 6\n3 7\n3 8\n', 'expected_output': '5\n1 8 6', 'factual_output': '4\n5 3 1'})
    thought = '''
We can use a stack to keep track of the operations and numbers as we parse the input. When we encounter an opening parenthesis, we push the current operation onto the stack and start a new operation. When we encounter a closing parenthesis, we pop the last operation from the stack and combine it with the current operation. When we encounter a number, we add it to the current operation. Finally, we evaluate the final operation to get the result.
'''

    for category, dataset in test_data.items():
        print(f"📂 处理类别: {category}，共 {len(dataset)} 道题目")

        # 使用 tqdm 显示进度条
        for row in tqdm(dataset, desc=f"Processing {category}"):
            if category != "competition" or row["id"] <= 3015 or row["id"] > 3029:
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
            #     "thought": thought,
            #     "best_code": code  # 也可以存储代码
            # }
            # save_result_entry(result_entry, RESULTS_FILE, is_first=first_write)
            # exit()
            # if category == "introductory" and row["id"] > 4029:
            #     continue
            # if category == "interview" and row["id"] > 29:
            #     continue
            # if category == "competition" and row["id"] > 3029:
            #     continue
            question_id = row["id"]
            solutions_list =[""]
            if row["solutions"] != "":# 个别题目没有solutions
                solutions_list = json.loads(row["solutions"])
            else:
                mcts.no_solutions = True
            # 在solutions非空的情况下，取第一个作为ground_truth_solution
            # 否则，将ground_truth_solution初始化为""，并在搜索过程中，始终将reward分数最高的代码作为ground_truth_solution
            mcts.ground_truth_solution = solutions_list[0]
            mcts.public_tests = row["public_tests"]
            mcts.private_tests = row["private_tests"]

            root = Node({"problem": row["question"], "thought": "", "code": "", "feedback": ""})
            # 初始root节点没有子节点，需要先进行扩展
            mcts.expand(root)

            start_time = time.time()
            # 运行 MCTS 进行代码生成
            for _ in range(10):  # 迭代次数
                leaf = mcts.select(root)
                if leaf.visits != 0:
                    leaf = mcts.expand(leaf)
                reward = mcts.evaluate(leaf)
                mcts.backpropagate(leaf, reward)
            elapsed = time.time() - start_time


            # 选择最佳代码并评估私有测试集
            # best_code = max(root.children, key=lambda x: x.pass_rate).state["code"]
            best_node = get_best_node(root)
            best_thought = best_node.state["thought"]
            best_code = best_node.state["code"]
            private_result = run_tests(best_code, row["private_tests"])

            # 记录结果
            result_entry = {
                "category": category,
                "question_id": question_id,
                "time_taken": fmt_duration(elapsed),
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