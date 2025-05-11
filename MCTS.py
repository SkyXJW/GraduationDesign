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
import sys

def solve(a, b, c):
    dp = [[[float('inf')] * (c + 1) for _ in range(b + 1)] for _ in range(a + 1)]
    dp[a][b][c] = 0
    for i in range(a, b + 1):
        for j in range(i, c + 1):
            for k in range(j, c + 1):
                if i == j == k:
                    dp[i][j][k] = 0
                else:
                    dp[i][j][k] = min(dp[i][j][k], dp[i][j - 1][k] + 1, dp[i][j + 1][k] + 1, dp[i - 1][j][k] + 1, dp[i + 1][j][k] + 1, dp[i][j][k - 1] + 1, dp[i][j][k + 1] + 1)
    res = float('inf')
    ans = (0, 0, 0)
    for i in range(a, b + 1):
        for j in range(i, c + 1):
            for k in range(j, c + 1):
                if j % i == 0 and k % j == 0:
                    if dp[i][j][k] < res:
                        res = dp[i][j][k]
                        ans = (i, j, k)
    return res, ans

t = int(input())
for _ in range(t):
    a, b, c = map(int, input().split())
    res, ans = solve(a, b, c)
    print(res)
    print(*ans)
'''
    code = '''
# Step 1:  Import the necessary libraries
from collections import defaultdict, deque

# Step 2:  Define a function to perform topological sorting on the graph
def topological_sort(graph, in_degree, n):
    # Step 3:  Initialize a queue with all nodes with in-degree 0
    
    queue = deque([i for i in range(1, n+1) if in_degree[i] == 0])
    # Step 4:  Initialize a list to store the sorted order of nodes
    
    sorted_order = []
    # Step 5:  Perform topological sorting
    
    while queue:
        node = queue.popleft()
        sorted_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    # Step 6:  Check if all nodes are included in the sorted order
    
    if len(sorted_order) == n:
        return sorted_order
    else:
        return []

# Step 7:  Define a function to find the minimum number of courses required to get the specialty
def min_courses(n, k, main_courses, dependencies):
    # Step 8:  Create a graph and in-degree dictionary
    
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    for i in range(1, n+1):
        in_degree[i] = 0
    # Step 9:  Add edges to the graph and update in-degree
    
    for i in range(1, n+1):
        for j in dependencies[i]:
            graph[j].append(i)
            in_degree[i] += 1
    # Step 10:  Perform topological sorting
    
    sorted_order = topological_sort(graph, in_degree, n)
    # Step 11:  Check if all main courses are included in the sorted order
    
    if len(sorted_order) == 0:
        return -1
    else:
        # Step 12:  Find the minimum number of courses required to get the specialty
        
        min_courses = set()
        for course in sorted_order:
            if course in main_courses:
                min_courses.add(course)
                for neighbor in graph[course]:
                    if neighbor not in min_courses:
                        min_courses.add(neighbor)
        return len(min_courses), min_courses

# Step 13:  Read input
n, k = map(int, input().split())
main_courses = set(map(int, input().split()))
dependencies = defaultdict(list)
for i in range(1, n+1):
    t = int(input())
    dependencies[i] = list(map(int, input().split()))
# Step 14:  Find the minimum number of courses required to get the specialty
min_courses, courses = min_courses(n, k, main_courses, dependencies)
# Step 15:  Print the result
if min_courses == -1:
    print(-1)
else:
    print(min_courses)
    print(*courses)
'''
    failed_cases = []
    failed_cases.append({'input': '8\n1 2\n2 3\n3 4\n4 5\n4 6\n3 7\n3 8\n', 'expected_output': '5\n1 8 6', 'factual_output': '4\n5 3 1'})
    current_thought = '''
The idea is to use Breadth-First Search (BFS) to find the longest path in the tree. We start BFS from an arbitrary vertex and find the farthest vertex from it. Then we start BFS from this farthest vertex and find the farthest vertex from it again. The path between these two farthest vertices is the longest path in the tree. We then choose three vertices from this path such that the number of edges which belong to at least one of the simple paths between them is the maximum possible. This can be achieved by choosing the first, middle, and last vertices of the path.
'''

    for category, dataset in test_data.items():
        print(f"📂 处理类别: {category}，共 {len(dataset)} 道题目")

        # 使用 tqdm 显示进度条
        for row in tqdm(dataset, desc=f"Processing {category}"):
            if category != "interview" or row["id"] > 29 or row["id"] <= 19:
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