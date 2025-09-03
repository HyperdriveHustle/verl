import re
from typing import Optional
from langdetect import detect_langs

# 数学变量白名单
MATH_WHITELIST = set(['x', 'y', 'z', 'a', 'b', 'c', 'd', 'n', 'm', 'k', 't', 's', 'r', 'p', 'q', 'f', 'g', 'h', 'i', 'j', 'l', 'u', 'v', 'w',
                      'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','(',')'])

# 英文专有名词白名单（合并数学变量）
EN_WHITELIST = set([
    'Answer'
]) | MATH_WHITELIST

# 常见英文单词（去掉数学常用词）
COMMON_EN_WORDS = set([
    'the', 'is', 'are', 'that', 'it', 'for', 'with', 'as', 'by', 'an', 'be', 'at'
])

def clean_text(text):
    # 去除LaTeX公式
    text = re.sub(r'\$\$.*?\$\$|\$.*?\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{([a-zA-Z*]+)\}.*?\\end{\1}', '', text, flags=re.DOTALL)
    # 去除数字和常见数学符号
    text = re.sub(r'[0-9+\-*/^_=(){}\[\]<>≤≥≠∑∏∫√π∞±∈∩∪]', '', text)
    # 去除数学变量和英文专有名词
    for w in EN_WHITELIST:
        text = re.sub(r'\b{}\b'.format(re.escape(w)), '', text)
    return text

def is_english_sentence(text):
    # 检查是否有完整英文句子（大写字母开头，句号结尾）
    return bool(re.search(r'[A-Z][a-zA-Z ,;:\'\"\-()]+[\.!?]', text))

def count_ch_en(text):
    ch_count = 0
    en_count = 0
    en_words = []
    for c in text:
        if '\u4e00' <= c <= '\u9fff':
            ch_count += 1
        elif 'a' <= c.lower() <= 'z':
            en_count += 1
    # 统计英文单词
    words = re.findall(r'[a-zA-Z]+', text)
    for w in words:
        if w not in EN_WHITELIST:
            en_words.append(w.lower())
    return ch_count, en_count, en_words

def detect_language(text):
    text_clean = clean_text(text)
    if not text_clean.strip():
        return "unknown"
    ch_count, en_count, en_words = count_ch_en(text_clean)
    total = ch_count + en_count
    en_word_set = set(en_words)

    score = 0

    # 1. 英文字符比例
    if total > 0 and en_count / total > 0.5:  
        score += 1

    # 2. 常见英文单词数量
    if len(en_word_set & COMMON_EN_WORDS) >= 8:  
        score += 1

    # 3. 是否有完整英文句子
    if is_english_sentence(text_clean):
        score += 1

    # 4. langdetect辅助
    try:
        lang_result = detect_langs(text_clean)
        zh_prob = 0.0
        en_prob = 0.0
        for res in lang_result:
            if 'zh' in res.lang:
                zh_prob = res.prob
            elif 'en' in res.lang:
                en_prob = res.prob
        if en_prob > 0.7:
            score += 1
        elif zh_prob > 0.7:
            score -= 1
    except:
        pass

    # 根据分数判定初步语言
    if score >= 2:
        # 初步判定为英文（包含纯英文和混合）
        # 进一步区分纯英文和混合
        # 这里定义：如果中文字符占比小于10%，判为纯英文，否则混合
        if total == 0:
            return "en"
        ch_ratio = ch_count / total
        if ch_ratio < 0.1:
            return "en"
        else:
            return "mix"
    else:
        return 'zh'