import math

def compute_score(solution_str, ground_truth):
    # print(type(solution_str),solution_str)
    # print("="*100)
    # scores, sub_reward = [], []
    # retval = 0.
    score=0
    try:
        format_score, answer_correct, cosine_reward, extracted_ans = evaluate_answer(solution_str, solution_str)
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

        # print({
        #     "score": score, 
        #     "gt": ground_truth, 
        #     "extracted_ans": extracted_ans
        # })
    except Exception as e:
        print(e)
    return score

def evaluate_answer(ans, ground_truth, is_base=True):
    '''判断答案和 gt 是否匹配'''
    # Initialize reward functions
    cosine_scaled_reward = get_cosine_scaled_reward(min_value_correct=0.5, max_value_correct=1.0, max_len=2048)
    
    extracted_ans = None
    follow_format, answer_with_boxed, answer_correct = 0, 0, 0
    # format_pattern = r'^<think>.*?<\/think>\s*<answer>.*?<\/answer>$'

    ans = ans.split("assistant:")
    ans = ans[1:]
    if len(ans) == 1: ans = ans[0]
    else: ans = "assistant:".join(ans)
    
    ans = ans.strip()
    # print(ans)
    # print("*"*100)
    # 0. 对于 base model 先判断是否遵循指令
    if ans.startswith("<think>") and ans.endswith("</answer>") and "<answer>" in ans  and "</think>" in ans:
        follow_format = 1
    # print(f">> follow_format = {follow_format}\n>> ans = {ans}")
    # 必须要 follow format 才可以继续训
    if follow_format > 0:
        # 1. 先判断时候包含 boxed
        extracted_ans = match_last_box_content(ans)
        print(f">> ans = {ans[-30:]}\n>> extracted_ans = {extracted_ans}")
        # 符合 \boxed{} 格式，能提取出来内容
        if extracted_ans is not None:
            answer_with_boxed = 1

        # 2. 根据 box 里面的内容判断
        if extracted_ans is not None:
            extracted_ans = remove_unnecessary(extracted_ans.strip())
            ground_truth = remove_unnecessary(ground_truth.strip())
            
            if len(extracted_ans) > 0:
                ans_set = parse_set(extracted_ans)
                gt_set = parse_set(ground_truth)
            
                # 完全匹配
                if ans_set == gt_set:
                    answer_correct = 1
                try:
                    ans_sympy = {sp.simplify(parse_latex(x)) for x in ans_set}
                    gt_sympy = {sp.simplify(parse_latex(x)) for x in gt_set}
                    if ans_sympy == gt_sympy:
                        # 数学表达式等价匹配
                        answer_correct = 1
                except:
                    pass
        
    # 添加 cos 长度惩罚
    cosine_reward = cosine_scaled_reward(ans, answer_correct == 1)
    return follow_format + answer_with_boxed, answer_correct, cosine_reward, extracted_ans

def get_cosine_scaled_reward(min_value_wrong: float = -1.0, max_value_wrong: float = -0.5,
                              min_value_correct: float = 0.5, max_value_correct: float = 1.0, max_len: int = 1000):
    """
    计算基于余弦调度的长度奖励
    """
    def cosine_scaled_reward(content, is_correct: bool) -> float:
        gen_len = len(content.split())
        progress = gen_len / max_len
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        return reward

    return cosine_scaled_reward

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