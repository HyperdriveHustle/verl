import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel,Field

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
    base_dir = "./tmp/ver1_firejail"

    # 2. 确保这个目录存在，如果不存在则创建它
    # exist_ok=True 确保了即使目录已经存在，这行代码也不会报错
    os.makedirs(base_dir, exist_ok=True)

    # 3. 现在可以安全地在已存在的目录中创建临时工作目录了
    workdir = tempfile.mkdtemp(prefix="fj_", dir=base_dir)
    script_path = os.path.join(workdir, "main.py")
    result = {
        "status": "unknown",
        "run_status": "unknown",
        "run_result": None
    }
    run_result = {}
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        cmd = [
            "firejail",
            f"--private={workdir}",              # 独立的工作目录
            f"--rlimit-as={req.memory_limit_MB * 1024 * 1024}",  # 内存限制 (Bytes)
            "--net=none",                        # 禁用网络
            "--private-dev",                     # 独立的 /dev
            "--seccomp",                         # 启用 seccomp 安全过滤器
            "--caps.drop=all",                   # 放弃所有 Linux capabilities
            "--noroot",                          # 禁用 root 权限
            "--quiet",                           # 静默模式，减少不必要的输出
            "--nosound",                         # 禁用声音设备
            req.language,                          # 使用从 language 映射来的可执行文件
            "main.py",                       # 要执行的脚本
        ]

        start_time = time.monotonic()
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=req.run_timeout, input=req.stdin
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
