import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def load_pairs(path: str, include_zero: bool = False):
    xs = []
    ys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            tp = obj.get("Training Possibility")
            ip = obj.get("Inference Possibility")
            if tp is None or ip is None:
                continue
            t = float(tp)
            x = float(ip)
            if not include_zero and (t == 0.0 or x == 0.0):
                continue
            ys.append(t)
            xs.append(x)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def compute_kl(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 50, eps: float = 1e-12) -> float:
    hist_p, edges = np.histogram(p_samples, bins=bins, range=(0.0, 1.0))
    hist_q, _ = np.histogram(q_samples, bins=edges)
    p = hist_p.astype(np.float64)
    q = hist_q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/afs/chatrl/users/zhr/models/verl_rl_models/verl_expert/GSPO-Async-test_2025-11-27_08-14-14/probability_data/token_probabilities.jsonl")
    p.add_argument("--output", default="probability_scatter.png")

    p.add_argument("--sample", type=int, default=100000)
    p.add_argument("--include_zero", action="store_true", default=True)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-12)
    args = p.parse_args()
    x, y = load_pairs(args.input, args.include_zero)
    if len(x) == 0:
        print("no_valid_pairs")
        return
    n = len(x)
    if args.sample > 0 and n > args.sample:
        rng = np.random.default_rng()
        idx = rng.choice(n, args.sample, replace=False)
        x = x[idx]
        y = y[idx]
    kl = compute_kl(y, x, args.bins, args.eps)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=2, alpha=0.3, rasterized=True)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Inference Probability")
    plt.ylabel("Training Probability")
    plt.text(0.02, 0.98, f"KL(Train||Infer)={kl:.4f}", transform=plt.gca().transAxes, ha="left", va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.6))
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(args.output)

if __name__ == "__main__":
    main()
