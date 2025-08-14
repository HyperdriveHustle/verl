import os
import shutil
import subprocess
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RunReq(BaseModel):
    code: str
    timeout: int = 90
    python_executable: str = "python3"

@app.post("/run")
def run(req: RunReq):
    workdir = tempfile.mkdtemp(prefix="fj_", dir="/tmp/verl_firejail")
    script_path = os.path.join(workdir, "main.py")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        cmd = [
            "firejail",
            f"--private={workdir}",
            "--quiet", "--net=none", "--private-dev", "--nosound", "--seccomp", "--caps.drop=all", "--noroot",
            req.python_executable, "main.py",
        ]
        proc = subprocess.run(
            cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=req.timeout
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return {"output": output, "status": "success" if proc.returncode == 0 else "failed", "returncode": proc.returncode}
    except subprocess.TimeoutExpired as e:
        return {"output": getattr(e, "stdout", "") or "" + getattr(e, "stderr", "") or str(e), "status": "failed", "returncode": None, "timeout": True}
    except Exception as e:
        return {"output": f"Error: {e}", "status": "failed", "returncode": None}
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
