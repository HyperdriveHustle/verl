# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Optional
from .language_detect import detect_language

try:
    import sympy as sp
    import re
    from sympy.parsing.latex import parse_latex
    from math_verify import parse, verify
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify sympy`.")



def remove_unnecessary(s):
    '''去掉不必要的符号和框
    '''
    for pattern in [
        "^\\circ", 
        "\\$", "\$", "\\%", "\%", " ", 
        "tfrac", "dfrac", "^{\\circ}",
        "\n", 
        "\\!"
    ]:
        s = s.replace(pattern, "")
    for item in ["^\\text", "\\mbox{", "\\text{", "^{\\text{"]:
        if len(s.split(item)) == 2:
            s = s.split(item)[0]
    return s


def match_answer_content(processed_str, answer_pattern = r'<answer>(.*?)</answer>'):
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        # print("verify <answer> not matches, return None")
        # print("[Error] No valid answer tags found")
        return None
    final_answer = matches[-1].group(1).strip()
    # print("verify <answer> matches, return final_answer")
    # print(final_answer)
    return final_answer

def convert_to_standard_number(s):
    try:
        return str(float(s)) if '.' in s or 'e' in s.lower() else str(int(s))
    except ValueError:
        return None

def parse_set(expr_str):
    elements = expr_str.split(',')
    parsed_elements = set()
    for elem in elements:
        elem = elem.strip()
        num = convert_to_standard_number(elem)
        if num is not None:
            parsed_elements.add(num)
        else:
            try:
                parsed_elements.add(str(parse_latex(elem)))
            except:
                parsed_elements.add(elem)
    return parsed_elements

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]

def compute_score(model_output: str, ground_truth: str) -> bool:
    # Limit solution length for efficiency
    response_str = model_output[-300:]  # The longest answer in MATH-500 has 159 characters
    # 按照 pattern 抽取出答案的部分
    extracted_ans = match_answer_content(
        response_str,
        # dapo prompt
        answer_pattern=r"(?i)Answer\s*:\s*([^\n]+)"
    )
    # 如果匹配不到Answer，则尝试匹配最后一个\boxed{...}
    if extracted_ans is None:
        boxed_str = last_boxed_only_string(response_str)
        if boxed_str is not None:
            extracted_ans = remove_boxed(boxed_str)
    # 根据 box 里面的内容判断（如果有 boxed 的话）
    format_correct= -1.0
    answer_correct = -1.0
    if extracted_ans is not None:
        format_correct = 1.0
        extracted_ans = remove_unnecessary(extracted_ans.strip())
        ground_truth = remove_unnecessary(ground_truth.strip())
        if len(extracted_ans) > 0:
            try:
                # math_verify判断
                answer_correct = 1.0 if verify(parse("$" + ground_truth + "$"), parse("$" + extracted_ans + "$")) else -1.0
                if answer_correct < 0:
                    ans_set = parse_set(extracted_ans)
                    gt_set = parse_set(ground_truth)
                    if ans_set == gt_set:
                        answer_correct = 1.0
                    else:
                        ans_sympy = {sp.simplify(parse_latex(x)) for x in ans_set}
                        gt_sympy = {sp.simplify(parse_latex(x)) for x in gt_set}
                        if ans_sympy == gt_sympy:
                            answer_correct = 1.0
            except:
                # print({"math_verify parse error": True, "extracted_ans": extracted_ans, "ground_truth": ground_truth})
                pass
    
    # print(f"[model_output] = \n{model_output}")
    # print(f"[ground_truth] = \n{ground_truth}")
    # print(f"[format_correct] = {format_correct}, [answer_correct] = \n{answer_correct}")
    # print("--"*10)
    
    # lang = detect_language(model_output) 
    # if lang != 'mix' and answer_correct == 1.0:
    #     reward = 1.0
    # elif lang == 'mix' and answer_correct == 1.0:
    #     reward = 0.2
    # elif lang != 'mix' and answer_correct <= 0:
    #     reward = -0.8
    # else:  # lang == 'mix' and answer_correct <= 0
    #     reward = -1.0

    # correct = 0 if reward <= 0 else 1
    # acc = correct
    # return {
    #     "score": reward,
    #     "acc": acc,
    #     "pred": "" if extracted_ans is None else extracted_ans,
    # }
    
    # correct 在 -1,1 之间
    if format_correct < 0:
        reward = format_correct
    else:
        reward = answer_correct
    
    # correct 在 0,1 之间
    # acc 在 0,1 之间
    correct = 0 if reward <= 0 else reward
    acc = correct
    # if lang == 'mix':
    #     # 如果是英文，直接惩罚
    #     return {
    #         "score": -1.0,
    #         "acc": acc,
    #         "pred": "" if extracted_ans is None else extracted_ans,
    #     } 
    # print(reward, acc, extracted_ans) 
    return {
            "score": reward,
            "acc": acc,
            "pred": "" if extracted_ans is None else extracted_ans,
        }