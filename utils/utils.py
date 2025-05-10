import subprocess
import ast
import sys
from io import StringIO
from prompt.template import *
import re
import json
import multiprocessing
from collections import deque

def generate_code(tokenizer, model, problem, thought):
#     thought="The idea is to use Breadth-First Search (BFS) to find the longest path in the tree. We start BFS from an arbitrary vertex and find the farthest vertex from it. Then we start BFS from this farthest vertex and find the farthest vertex from it again. The path between these two farthest vertices is the longest path in the tree. We then choose three vertices from this path such that the number of edges which belong to at least one of the simple paths between them is the maximum possible. This can be achieved by choosing the first, middle, and last vertices of the path."
#     thought='''
# 1. **Identify the Longest Path**: The longest path in a tree is known as its diameter. We can find this using two Breadth-First Searches (BFS). The first BFS from an arbitrary node finds the farthest node from it. The second BFS from this farthest node gives the diameter of the tree.
# 2. **Select Vertices on the Longest Path**: Once we have the diameter, we select three vertices such that they are the first, middle, and last nodes of this path. This selection ensures that the union of the paths between these vertices covers the maximum number of edges.
# '''
    patterns = [
        r"can't be solved by Python",
        r"not solvable by Python",
        r"impossible to solve.*Python",
        r"Python.*can't solve",
        r"no solution in Python"
    ]
    if not thought: # 实际上这里的thought永不为空
        print(code_generation_no_thought.format(problem))
        inputs = tokenizer(code_generation_no_thought.format(problem),return_tensors="pt")
    else:
        # 由于在实际实验过程中发现，LLM Agent给出的thought还可能为"The problem can't be solved by Python."之类的
        # 为此直接返回"", ""
        print(code_generation_with_thought.format(problem,thought))
        if any(re.search(pattern, thought, re.IGNORECASE) for pattern in patterns):
            return "", ""
        inputs = tokenizer(code_generation_with_thought.format(problem,thought),return_tensors="pt")
    outputs = model.generate(input_ids=inputs.input_ids.cuda(), 
                    attention_mask=inputs.attention_mask.cuda(),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=100000,
                    num_return_sequences=1) # 设置输出回答数目为3条
    response = [tokenizer.decode(output[inputs.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for output in outputs]
    print(response[0])
    # print("response[0]:\n"+response[0])

    match = re.search(r"<code>(.*?)</code>|```python\s+([\s\S]*?)```", response[0], re.DOTALL)
    if match:
        code = match.group(1) if match.group(1) else match.group(2)
    else:
        # 实际实验过程中发现，LLM给出的代码也会直接裸露在外，不被包含在任何标签中
        code = response[0].strip()
        # raise ValueError("Failure to capture valid code")
    if not thought:
        match = re.search(r"<idea>(.*?)</idea>|###\s*Approach\s+(.*?)###\s*Solution Code", response[0], re.DOTALL)
        if match:
            thought = match.group(1) if match.group(1) else match.group(2).strip()
        else:
            raise ValueError("Failure to capture valid thought")
    # print("code:\n"+code)
    return thought, code

# 由于LLM给出的回答中，除了最外层的[]，还会包含'['、']'，导致会出现二义性，这里单独处理
def extract_json(text):
    start = text.find('[')
    if start == -1:
        raise ValueError("No opening bracket found")
    count = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            count += 1
        elif text[i] == ']':
            count -= 1
            if count == 0:
                return text[start:i+1]
    raise ValueError("No matching closing bracket found")
# 由于LLM给出的回答中，会出现""，导致json.loads()解析失败，所以这里进行手动解析
def manual_json_parse(s):
    # 预处理：去头尾括号、换行和空格
    s = re.sub(r'[\n\t]', '', s.strip()[1:-1])
    
    records = []
    # 分割记录（兼容最后一条无逗号的情况）
    for raw_record in re.split(r'\},\s*', s):
        if not raw_record: continue
        record = {}
        # # 提取键值对
        # for match in re.finditer(r'"([^"]+)":\s*("[^"]*"|\d+\.?\d*)', raw_record):
        #     key = match.group(1)
        #     value = match.group(2)
        #     # 类型转换
        #     if value.startswith('"'):
        #         value = value[1:-1].replace('\\"', '"')
        #     else:
        #         value = float(value) if '.' in value else int(value)
        #     record[key] = value
        # 上述的re正则匹配的方式不太适用，于是选择以下较为暴力、笨拙的方法
        # 提取 Thought
        k1 = raw_record.find('"Thought":')
        if k1 != -1:
            start = raw_record.find('"', k1 + 9) + 1
            # end = start
            # while end < len(raw_record):
            #     # if raw_record[end] == '"' and raw_record[end - 1] != '\\':
            #     if end == k-2:
            #         break
            #     end += 1
            end = raw_record.find('"Reasonableness":')
            while raw_record[end] != ',':
                end = end - 1
            record["Thought"] = raw_record[start:end-1]

        # 提取 Reasonableness
        k2 = raw_record.find('"Reasonableness":')
        if k2 != -1:
            start = k2 + len('"Reasonableness":')
            while start < len(raw_record) and raw_record[start] in ' \t':
                start += 1
            end = start
            while end < len(raw_record) and (raw_record[end].isdigit() or raw_record[end] in '.'):
                end += 1
            val = float(raw_record[start:end]) if '.' in raw_record[start:end] else int(raw_record[start:end])
            record["Reasonableness"] = val
        records.append(record)
    return records

def generate_thought(tokenizer, model, problem, thought, code, feedback, k):
    if not feedback: # 这里说明当前扩展的node是root,此时仅有problem
        inputs = tokenizer(thought_generation_no_feedback.format(problem,k), return_tensors="pt")
        print(thought_generation_no_feedback.format(problem,k))
    else:
        inputs = tokenizer(thought_generation_with_feedback.format(code,feedback,k,problem,thought), return_tensors="pt")
        print(thought_generation_with_feedback.format(code,feedback,k,problem,thought))

    outputs = model.generate(input_ids=inputs.input_ids.cuda(), 
                     attention_mask=inputs.attention_mask.cuda(),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=100000, 
                    num_return_sequences=1)
    new_thoughts = [tokenizer.decode(output[inputs.input_ids.size(1):].cpu(),
                         skip_special_tokens=True).strip() for output in outputs]
    # print(new_thoughts[0])
    
    json_str = extract_json(new_thoughts[0])
    print(manual_json_parse(json_str))
    return manual_json_parse(json_str)
    # return json.loads(json_str)

    # match = re.search(r"\[\s*(.*?)\s*\]", new_thoughts[0], re.DOTALL)
    # if match:
    #     json_str = "[" + match.group(1) + "]"
    #     try:
    #         return json.loads(json_str)
    #     except json.JSONDecodeError as e:
    #         raise ValueError("JSON parsing error:", e)
    # else:
    #     raise ValueError("Failure to capture valid thought")

def run_code_in_process(code, input_data, timeout=2):
    # 在子进程中运行给定代码，使用指定的timeout秒数限制执行时间。
    # 如果超时则返回None，否则返回程序输出
    def target(result_queue):
        try:
            # 重定向输入输出
            sys.stdin = StringIO(input_data)
            sys.stdout = StringIO()
            
            # 执行代码，并将环境放到字典中
            exec_env = {}
            exec(code, exec_env)
            
            # 如果代码中定义了 main(), 则调用它
            if "main" in exec_env:
                exec_env["main"]()
            
            # 将标准输出的内容放入队列返回
            result_queue.put(sys.stdout.getvalue().strip())
        except Exception as e:
            result_queue.put(f"Error: {e}")

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()
    
    # 等待规定时间，timeout秒内未结束认为超时
    process.join(timeout)
    if process.is_alive():
        process.terminate()  # 杀死进程
        process.join()
        return None   # 用 None 来表示超时（或死循环）
    
    try:
        return result_queue.get_nowait()
    except Exception:
        return None

# def run_tests(code, test_cases):
#     try:
#         passed = 0
#         failed_cases = []

#         for test in test_cases:
#             input_data = test["input"]
#             expected = test["output"]

#             try:
#                 # 设置 `sys.stdin` 以模拟输入
#                 sys.stdin = StringIO(input_data)
#                 sys.stdout = StringIO()  # 捕获 `print()` 输出

#                 exec_env = {}
#                 exec(code, exec_env)  # 运行代码

#                 # 如果代码里有 `main()`，调用 `main()`
#                 if "main" in exec_env:
#                     exec_env["main"]()

#                 # 获取 `print()` 的最终输出
#                 result = sys.stdout.getvalue().strip()

#                 # sys.stdout = sys.__stdout__
#                 # print(result)
#                 # print(expected.strip())

#                 # 比较输出
#                 if result == expected.strip():
#                     passed += 1
#                 else:
#                     failed_cases.append({
#                         "input": input_data,
#                         "expected_output": expected.strip(),
#                         "factual_output": result
#                     })

#             except Exception as e:
#                 failed_cases.append({
#                     "input": input_data,
#                     "expected_output": expected.strip(),
#                     "factual_output": f"Error: {e}"
#                 })
            
#             finally:
#                 # 还原 `sys.stdin` 和 `sys.stdout`
#                 sys.stdin = sys.__stdin__
#                 sys.stdout = sys.__stdout__

#         return {
#             "pass_rate": passed / len(test_cases),
#             "failed_cases": failed_cases,
#             "error": None
#         }

#     except Exception as e:
#         return {
#             "pass_rate": 0.0,
#             "failed_cases": [],
#             "error": str(e)
#         }

def run_tests(code, test_cases):
    # 对传入的代码和测试用例进行测试，
    # 如果测试代码在规定时间内没有结束（如陷入死循环），将该次测试视为失败并记录
    try:
        passed = 0
        failed_cases = []
        timeout = 2  # 设置每个测试用例允许的最大执行时间（秒）
    
        for test in test_cases:
            input_data = test["input"]
            expected = test["output"].strip()
    
            result = run_code_in_process(code, input_data, timeout)
            
            # 判断是否超时或死循环
            # 如果超时或死循环，则将此测试用例的factual_output输出记录为空
            if result is None:
                failed_cases.append({
                    "input": input_data,
                    "expected_output": expected,
                    "factual_output": ""
                })
            else:
                if result == expected:
                    passed += 1
                else:
                    failed_cases.append({
                        "input": input_data,
                        "expected_output": expected,
                        "factual_output": result
                    })
    
        # # 如果任何测试用例超时导致的失败，也可根据要求选择直接返回 pass_rate 为 0 
        # # 例如：若有任一测试用例超时，则直接认为整体执行失败：
        # for case in failed_cases:
        #     if "Timeout" in case["factual_output"]:
        #         return {
        #             "pass_rate": 0.0,
        #             "failed_cases": failed_cases,
        #             "error": "Detected an infinite loop or timeout."
        #         }
    
        return {
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "failed_cases": failed_cases,
            "error": None
        }
    
    except Exception as e:
        return {
            "pass_rate": 0.0,
            "failed_cases": [],
            "error": str(e)
        }


def self_assess(tokenizer, model, problem, code):
    # print(code_evaluation.format(problem, code))
    inputs = tokenizer(code_evaluation.format(problem, code),return_tensors="pt")
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

class Normalizer(ast.NodeTransformer):
    """归一化AST节点，将变量名和常量值统一"""
    def visit_Name(self, node):
        return ast.Name(id='var', ctx=node.ctx)
    
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return ast.Constant(value=0)
        elif isinstance(node.value, str):
            return ast.Constant(value="")
        elif isinstance(node.value, bool):
            return ast.Constant(value=True)
        else:
            return ast.Constant(value=None)

def parse_and_normalize(code):
    """解析代码并返回归一化的AST"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    normalizer = Normalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    return normalized_tree

def ast_to_tree(node):
    """将AST节点转换为(类型, 子节点列表)的结构"""
    if isinstance(node, ast.AST):
        node_type = type(node).__name__
        children = []
        for field in node._fields:
            value = getattr(node, field)
            if isinstance(value, list):
                for item in value:
                    children.append(ast_to_tree(item))
            else:
                children.append(ast_to_tree(value))
        return (node_type, children)
    elif isinstance(node, list):
        return [ast_to_tree(item) for item in node]
    else:
        return node

def sim_ast(tree1, tree2):
    """递归计算两个AST树的相似度"""
    if tree1 == tree2:
        return 1.0
    if isinstance(tree1, tuple) and isinstance(tree2, tuple):
        if tree1[0] != tree2[0]:
            return 0.0
        children1 = tree1[1] if isinstance(tree1[1], list) else []
        children2 = tree2[1] if isinstance(tree2[1], list) else []
        children_score = 0.0
        for c1, c2 in zip(children1, children2):
            children_score += sim_ast(c1, c2)
        len1, len2 = len(children1), len(children2)
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        return children_score / max_len
    return 0.0

def compare_code_similarity(code1, code2):
    """比较两段代码的AST相似度"""
    tree1 = parse_and_normalize(code1)
    tree2 = parse_and_normalize(code2)
    if tree1 is None or tree2 is None:
        return 0.0
    struct1 = ast_to_tree(tree1)
    struct2 = ast_to_tree(tree2)
    return sim_ast(struct1, struct2)

def get_best_node(root):
    """
    BFS 遍历整棵树，返回 pass_rate 最大节点
    """
    queue = deque([root])
    best_node = root

    while queue:
        node = queue.popleft()
        if node.pass_rate > best_node.pass_rate:
            best_node = node
        elif node.pass_rate == best_node.pass_rate:
            # 由于树中会存在多个没有thought或code，而pass_rate又相等的节点
            # 所以这里需要在pass_rate相同时，提前判断code是否为空
            if node.state["thought"] != "":
                if node.state["code"] == "":
                    if best_node.state["thought"] == "":
                        best_node = node
                else:
                    best_node = node
        for c in node.children:
            queue.append(c)

    return best_node

def fmt_duration(seconds: float) -> str:
    total_sec = int(seconds)
    minutes = total_sec // 60
    secs = total_sec % 60
    return f"{minutes}m{secs}s"

def save_result_entry(result_entry, RESULTS_FILE, is_first=False):
    """ 追加单条记录到 JSON 文件 """
    mode = "w" if is_first else "a"  # 第一次写入时使用 'w'，否则用 'a'
    with open(RESULTS_FILE, mode, encoding="utf-8") as f:
        if is_first:
            f.write("[\n")  # 首次写入，创建 JSON 数组
        else:
            f.write(",\n")  # 追加模式，加逗号分隔
        json.dump(result_entry, f, indent=4, ensure_ascii=False)

def finalize_results(RESULTS_FILE):
    """ 补全 JSON 文件（加上 `]`）"""
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n]")  # 关闭 JSON 数组
# 示例用法
if __name__ == "__main__":
    # code1 = "x = a + 1"
    # code2 = "y = b + 2"
    # print(compare_code_similarity(code1, code2))  # 输出1.0

    # code3 = "def foo(): pass"
    # code4 = "def bar(): pass"
    # print(compare_code_similarity(code3, code4))  # 输出1.0

    # code5 = "x = a + 'hello'"
    # code6 = "y = b + 3"
    # print(compare_code_similarity(code5, code6))  # 输出0.0
    code = """
import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    n = int(stdin.readline())
    edges = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a, b = map(int, stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    
    max_depth = [0] * (n + 1)
    farthest_node = [0] * (n + 1)
    best_sum = 0
    best_a, best_b, best_c = -1, -1, -1
    
    stack = [(1, -1, False)]
    
    while stack:
        node, parent, visited = stack.pop()
        if not visited:
            stack.append((node, parent, True))
            for neighbor in reversed(edges[node]):
                if neighbor != parent:
                    stack.append((neighbor, node, False))
        else:
            children = []
            for neighbor in edges[node]:
                if neighbor != parent:
                    children.append(neighbor)
            children_distances = []
            for child in children:
                d = 1 + max_depth[child]
                node_child = farthest_node[child]
                children_distances.append((d, node_child))
            children_distances.sort(reverse=True, key=lambda x: x[0])
            
            sum_d = 0
            a, b, c = -1, -1, -1
            count = 0
            for i in range(min(3, len(children_distances))):
                sum_d += children_distances[i][0]
                if count == 0:
                    a = children_distances[i][1]
                elif count == 1:
                    b = children_distances[i][1]
                elif count == 2:
                    c = children_distances[i][1]
                count += 1
            
            if children_distances:
                max_d = children_distances[0][0]
                farthest_node[node] = children_distances[0][1]
                max_depth[node] = max_d
            else:
                max_depth[node] = 0
                farthest_node[node] = node
            
            if a != -1 and b != -1 and c != -1:
                if a != b and b != c and a != c:
                    if sum_d > best_sum:
                        best_sum = sum_d
                        best_a, best_b, best_c = a, b, c
                else:
                    if len(children_distances) >= 2:
                        a, b = children_distances[0][1], children_distances[1][1]
                        if a != b:
                            if sum_d > best_sum:
                                best_sum = sum_d
                                best_a, best_b, best_c = a, b, node
                    elif len(children_distances) == 1:
                        a = children_distances[0][1]
                        if sum_d > best_sum:
                            best_sum = sum_d
                            best_a, best_b, best_c = a, node, node
            else:
                if len(children_distances) >= 1:
                    a = children_distances[0][1]
                    if sum_d > best_sum:
                        best_sum = sum_d
                        best_a, best_b, best_c = a, node, node
    
    if best_a == -1 or best_b == -1 or best_c == -1:
        best_a, best_b, best_c = 1, 2, 3
    
    print(best_sum)
    print(best_a, best_b, best_c)

if __name__ == "__main__":
    main()
        """
    test_cases = [
        {"input": "8\n1 2\n2 3\n3 4\n4 5\n4 6\n3 7\n3 8\n",
         "output": "5\n1 8 6\n"}
    ]
    print(run_tests(code,test_cases))