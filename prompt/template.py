# Qwen生成的response格式为<code> </code>\n<idea> </idea>
# DeepSeek生成的response格式为### Approach\n### Solution Code ```python ```\n### Explanation
code_generation_no_thought = "Problem:\n{}\n"\
                  "Above is a Python code problem. "\
                  "Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response. "\
                  "No other explanation or words attached in your code! After generating the code, you should also give the corresponding code writing ideas. "\
                  "Here is the template of response:\n"\
                  "<code> your code here </code>\n<idea> code writing idea here </idea>.\n"

# Qwen生成的response格式为```python ```
# DeepSeek生成的response格式为### Solution Code ```python ```\n### Explanation
code_generation_with_thought = "Following the given coding ideas below, please complete the corresponding Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response. "\
                  "Generate the code ONLY. No other explanation or words attached! "\
                  "The problem is described below:\n{}\n"\
                  "The coding idea is as follows:\n{}\n"

# Qwen与DeepSeek生成的response格式为evaluation: xx
# Qwen生成的response格式还可能为xxxx. The correctness score is x.
code_evaluation = "You are a evaluator that evaluates the code is suitable for solving a given problem.\n" \
                  "Problem:\n{}\nCode:\n{}" \
                  "Above is a Python code problem and code to solve the problem. The code could pass all the example test cases, however, it may or may not be completely correct. " \
                  "Please evaluate and return the correctness score in range [-1, 1]. Evaluate the correctness of the code and give only ONE evaluation score. The code's correctness is whether it can pass all the possible unseen test cases of the problem, not just the given ones. " \
                  "Here are some Examples:\n" \
                  "evaluation: -0.5,  explanation: The code is far from correct for solving the problem. " \
                  "evaluation: 1.0, explanation: The generated code is the correct solution that can pass all the possible test cases and strange corner cases too. " \
                  "evaluation: 0.1, explanation: The code is not the correct solution but can pass some simple test cases. " \
                  "evaluation: 0.85, explanation: The code can pass most test cases while may fail on some corner cases. " \
                  "After evaluating the code, you should return the score, then provide an explanation.\n"

# Qwen与DeepSeek生成的reponse格式为[{"Thought": "xx", "Reasonableness": xx}]
thought_generation_no_feedback = "I need you analyze the following problem and provide solving strategies. Here is the problem description:\n{}\n" \
                            "I need you to output {} possible thoughts and strategies. Remember each only contain one possible strategy of the problem. "\
                            "Please wrap your response into a JSON object that contains keys `Thought`, and key `Reasonableness` with the Reasonableness of each thought. "\
                            "The JSON should be a **list of dicts**, the dicts are split with comma ','. "\
                            "Example Answers:\n" + '''
[
    {{"Thought":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7}},
    {{"Thought":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29}},
    {{"Thought":" The problem can't be solved by Python.", "Reasonableness": 0.01}}
]
'''

# Qwen与DeepSeek生成的reponse格式为[{"Thought": "xx", "Reasonableness": xx}]
thought_generation_with_feedback = "Code:\n{}\nFailed test cases:\n{}\n" \
                                   "The above code generated failed on some test cases. The errors or in-correct outputs are also given above. "\
                                   "I need you to analyze and modify the current thought for the programmer that can lead to the correct solution code. "\
                                   "The goal is that the modified thought could lead to the code that not only avoids the current error but also solve the problem in a way that handles other potential test cases that we haven't encountered yet. "\
                                   "I need you to output {} possible modified thoughts. Remember each only contain one possible strategy after modifying to correct the error and solve the problem. "\
                                   "Please wrap your response into a JSON object that contains keys `Thought`, and key `Reasonableness` with the Reasonableness of each thought. "\
                                   "The JSON should be a **list of dicts**, the dicts are split with comma ','. "\
                                   "Example Answers:\n" + '''
[
    {{"Thought":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7}},
    {{"Thought":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29}},
    {{"Thought":" The problem can't be solved by Python.", "Reasonableness": 0.01}}
]
''' + "\nHere is the problem description and current thought:\nProblem:\n{}\nCurrent Thought:\n{}\n"\
"Remember to strictly follow the format of the above example answers in your responses and each strategy after modifying should be not the same as each other.\n"

# 在消融实验部分，调用DeepSeek-R1对给定的thought进行评分
# 此外在TOT方法中，需要模型对thought进行评估打分
thought_evaluation = "Following is a Python code problem:\n{}\n"\
                    "Given the following code writing idea, I need you to give it a comprehensive rating of [0,1] in terms of time complexity, space complexity, and whether it passes all potential test cases."\
                    "Here are some Examples:\n" \
                    "evaluation: 0.0, explanation: The thought is far from correct for solving the problem. " \
		            "evaluation: 0.2, explanation: The time complexity of this thought is too high for solving the problem. " \
		            "evaluation: 0.2, explanation: The space complexity of this thought is too high for solving the problem. " \
                    "evaluation: 1.0, explanation: The thought is the correct solution that can pass all the possible test cases and strange corner cases too. " \
                    "evaluation: 0.1, explanation: The thought is not the correct solution but can pass some simple test cases. " \
                    "evaluation: 0.85, explanation: The thought can pass most test cases while may fail on some corner cases. " \
                    "The coding idea is as follows:\n{}\n"


# TOT
# Qwen与DeepSeek生成的reponse格式为[{"Thought": "xx"}]
tot_thought_generation_no_thought = "I need you analyze the following problem and provide solving strategies. Here is the problem description:\n{}\n" \
                            "I need you to output {} possible thoughts and strategies. Remember each only contain one possible strategy of the problem. "\
                            "Please wrap your response into a JSON object that contains keys `Thought`. "\
                            "The JSON should be a **list of dicts**, the dicts are split with comma ','. "\
                            "Example Answers:\n" + '''
[
    {{"Thought":" We could use the print function to finish the task in one line: print(2 + 3)"}},
    {{"Thought":" We should calculate the problem by setting a=2+3, and then print(a)"}},
    {{"Thought":" The problem can't be solved by Python."}}
]
'''

# Qwen与DeepSeek生成的reponse格式为[{"Thought": "xx", "Reasonableness": xx}]
tot_thought_generation_with_thought = "I need you to analyze and modify the current thought for the programmer that may lead to the correct solution code. "\
                                   "The goal is that the modified thought could lead to the code that not only avoids the current error but also solve the problem in a way that handles other potential test cases that we haven't encountered yet. "\
                                   "I need you to output {} possible modified thoughts. Remember each only contain one possible strategy after modifying to correct the error and solve the problem. "\
                                   "Please wrap your response into a JSON object that contains keys `Thought`. "\
                                   "The JSON should be a **list of dicts**, the dicts are split with comma ','. "\
                                   "Example Answers:\n" + '''
[
    {{"Thought":" We could use the print function to finish the task in one line: print(2 + 3)"}},
    {{"Thought":" We should calculate the problem by setting a=2+3, and then print(a)"}},
    {{"Thought":" The problem can't be solved by Python."}}
]
''' + "\nHere is the problem description and current thought:\nProblem:\n{}\nCurrent Thought:\n{}\n"\

# Base
base_prompt = "Problem:\n{}\n"\
            "Above is a Python code problem. "\
            "Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response. "\
            "No other explanation or words attached in your code! Here is the template of response:\n"\
            "<code> your code here </code>.\n"

# COT
cot_prompt = "Problem:\n{}\n"\
            "Above is a Python code problem. "\
            "Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response. "\
            "Make a coding idea then generate th code. No other explanation or words attached in your code! "\
            "Here is the template of response:\n"\
            "<idea> code writing idea here. </idea>\n<code> your code here. </code>"

