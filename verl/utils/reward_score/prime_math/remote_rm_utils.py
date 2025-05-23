import time
import ray
import requests
import torch
import re
import json
import math
import sympy as sp
from sympy.parsing.latex import parse_latex
from math_verify import parse, verify
from .logging_utils import init_logger

logger = init_logger(__name__)




def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def match_last_box_content(s):
    '''取出 \box{} 里面的内容
    '''
    box_matches = [m.start() for m in re.finditer(r'\\boxed\{', s)]
    if not box_matches:
        return None  # 不符合 \boxed{} 格式
    last_box_start = box_matches[-1]
    stack = ['{']
    result = None
    i = last_box_start + len(r'\boxed{')
    while i < len(s):
        if s[i] == '{':
            stack.append(i)
        elif s[i] == '}':
            stack.pop()
            if not stack:
                result = s[last_box_start + len(r'\boxed{'):i]
                break
        i += 1
    return result

def match_answer_content(processed_str):
    # print("verify match_answer_content in")
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        # print("verify <answer> not matches, return None")
        # print("[Error] No valid answer tags found")
        return None
    final_answer = matches[-1].group(1).strip()
    # print("verify <answer> matches, return final_answer")
    # print(final_answer)
    return final_answer

def remove_unnecessary(s):
    '''去掉不必要的符号和框
    '''
    for pattern in ["^\\circ", "\\$", "\$", "\\%", "\%", " ", "tfrac", "dfrac", "^{\\circ}","\n","\\!"]:
        s = s.replace(pattern, "")
    for item in ["^\\text", "\\mbox{", "\\text{", "^{\\text{"]:
        if len(s.split(item)) == 2:
            s = s.split(item)[0]
    return s


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


def evaluate_answer(ans, ground_truth, is_base=True):
    '''判断答案和 gt 是否匹配'''
    
    extracted_ans = None
    follow_format, answer_with_boxed, answer_correct = 0, 0, 0
    
    if ground_truth.startswith("$") and ground_truth.endswith("$"):
        ground_truth = ground_truth[1:-1].strip()

    if "<answer>" in ans:
        follow_format = 1
    # 必须要 follow format 才可以继续训
    if follow_format > 0:
        # 1. 先判断时候包含 boxed
        extracted_ans = match_answer_content(ans)
        # print(f">> ans = {ans[-30:]}\n>> extracted_ans = {}")
        # 符合 \boxed{} 格式，能提取出来内容
        if extracted_ans is not None:
            answer_with_boxed = 1

        # 2. 根据 box 里面的内容判断
        if extracted_ans is not None:
            # print({"extracted_ans_before":extracted_ans})
            extracted_ans = remove_unnecessary(extracted_ans.strip())
            ground_truth = remove_unnecessary(ground_truth.strip())
            # print({"ground_truth":ground_truth})
            # print({"extracted_ans_after":extracted_ans})
            if len(extracted_ans) > 0:
                try:
                    #math_verify判断
                    answer_correct = 1 if verify(parse("$" + ground_truth + "$"), parse("$" + extracted_ans + "$")) else 0

                    if answer_correct == 0:
                        ans_set = parse_set(extracted_ans)
                        gt_set = parse_set(ground_truth)
                        if ans_set == gt_set:
                            answer_correct = 1
                        else:
                            ans_sympy = {sp.simplify(parse_latex(x)) for x in ans_set}
                            gt_sympy = {sp.simplify(parse_latex(x)) for x in gt_set}
                            if ans_sympy == gt_sympy:
                                answer_correct = 1            
                except:
                    pass
        
    if answer_correct == 1:
        answer_correct = True
    else:
        answer_correct = False
    # return answer_correct
    return follow_format + answer_with_boxed, answer_correct, extracted_ans


def remote_rm_math_fn(api_url, queries, ground_truths, score_key="rewards", is_base=True):
    """remote reward model API
    api_url: multi-reward-api conbined by ,
    queries: query+response with the template
    design is made optional.
    ground_truths: answer
    score_key: RM score key
    """
    scores, sub_reward = [], []
    for query, gt in zip(queries, ground_truths):
        format_score, answer_correct, cosine_reward, extracted_ans = evaluate_answer(query, gt, is_base=is_base)
        # 格式完全不对
        if format_score == 0:
            score = -1
        else:
            # 如果能提取 box 的内容，则按照 cos 判断
            score = 0
            if "format" in api_url:
                score += format_score * 0.1
            if "answer_correct" in api_url:
                score += answer_correct
            if "answer_pos_neg" in api_url:
                score += answer_correct if answer_correct == 1 else -1
            if "cosine" in api_url:
                score += cosine_reward

        print({
            "score": score, 
            "gt": gt, 
            "extracted_ans": extracted_ans
        })
        scores.append(score)
        sub_reward.append(dict(format_score=format_score, answer_correct=answer_correct, cosine_reward=cosine_reward))
    return (torch.tensor(scores), sub_reward)


def remote_rm_fn(api_url, queries, ground_truths, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    # scores = request_api_wrapper(api_url, {"query": queries}, score_key)    
    scores = []
    for query, gt in zip(queries, ground_truths):
        # 从query末尾提取[[A]]或[[B]]
        match = re.findall(r'\[\[(A|B)\]\]', query)
        if match:
            extracted = match[-1]
            # 比较提取的值和ground truth是否一致
            score = 1.0 if extracted == gt else 0.0
            scores.append(score)
        else:
            scores.append(0.0)
        
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, ground_truths, score_key="rewards"):
    return remote_rm_math_fn(api_url, queries, ground_truths, score_key)
    # return remote_rm_fn(api_url, queries, ground_truths, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
