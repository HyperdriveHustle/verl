import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel,Field

PY_IMPORTS = """import heapq
import itertools
import random
import functools
import collections
import string
import math
import datetime

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
            i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
            i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

"""

app = FastAPI()

class RunReq(BaseModel):
    code: str
    language: str
    run_timeout: int = 30 
    compile_timeout: int = 10
    stdin: Optional[str] = ""
    memory_limit_MB: int = 128

    files: Optional[Dict[str, str]] = Field(default_factory=dict)
    fetch_files: Optional[List[str]] = Field(default_factory=list)

@app.post("/run")
def run(req: RunReq):
    #base_dir = "/tmp/verl_firejail"

    # 2. 确保这个目录存在，如果不存在则创建它
    # exist_ok=True 确保了即使目录已经存在，这行代码也不会报错
    #os.makedirs(base_dir, exist_ok=True)

    # 3. 现在可以安全地在已存在的目录中创建临时工作目录了
    try:
        with tempfile.TemporaryDirectory(prefix="ver1_fj_", dir="/tmp") as workdir:
            script_path = os.path.join(workdir, "main.py")
            result = {
                "status": "unknown",
                "run_status": "unknown",
                "run_result": None
            }
            code = PY_IMPORTS + req.code
            run_result = {}
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = [
                "firejail",
                f"--private={workdir}",              # 独立的工作目录
                f"--rlimit-as={req.memory_limit_MB}m",  # 内存限制 (Bytes)
                "--rlimit-fsize=2m",  # Limit file size
                "--seccomp",                         # 启用 seccomp 安全过滤器
                "--quiet",                           # 静默模式，减少不必要的输出
                f"--timeout=00:00:{req.run_timeout}",
                f"--whitelist={workdir}",
                req.language,                          # 使用从 language 映射来的可执行文件
                "main.py",                       # 要执行的脚本
            ]

            start_time = time.monotonic()
            proc = subprocess.run(
                cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True, 
                timeout=req.run_timeout, input=req.stdin
            )
            duration = time.monotonic() - start_time

        run_result["stdout"] = proc.stdout
        run_result["stderr"] = proc.stderr
        run_result["return_code"] = proc.returncode
        run_result["execution_time"] = duration
        result["run_result"] = run_result
        
        # 7. 根据退出码衍生状态
        if proc.returncode == 0:
            result["status"] = "Success"
            run_result["status"] = "Finished"
        else:
            result["status"] = "Failed"
            run_result["status"] = "Finished"

        result["run_result"] = run_result
        return result
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - start_time
        result["status"] = "Failed"
        run_result["status"] = "TimeLimitExceeded"
        run_result["stderr"] = "TimeLimitExceeded"
        run_result["execution_time"] = duration
        result["run_result"] = run_result
        return result
        
    except Exception as e:
        result["status"] = "Failed"
        run_result["status"] = "Error"
        run_result["stderr"] = f"An unexpected internal error occurred: {e}"
        result["run_result"] = run_result
        return result
        
    finally:
        if 'workdir' in locals() and os.path.exists(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
