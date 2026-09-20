import time
import torch
import torch.nn as nn
import numpy as np

def benchmark():
    print("================================================================================")
    print(" ITEM 8: CHUNKED PER-SAMPLE GRADIENTS BENCHMARK (B=300, 1000, 2000)")
    print("================================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use realistic HVAE-sized architecture (~150k params)
    in_dim = 72
    joint_dim = 192
    decoder = nn.Sequential(
        nn.Linear(joint_dim, 128),
        nn.SiLU(),
        nn.Linear(128, 256),
        nn.SiLU(),
        nn.Linear(256, in_dim),
    ).to(device)
    params = list(decoder.parameters())
    num_params = sum(p.numel() for p in params)
    print(f"Model parameters: {num_params:,}")

    C = 0.32
    chunk_size = 128
    batch_sizes = [300, 1000, 2000]

    results = []

    for B in batch_sizes:
        print(f"\n--- Testing Batch Size B = {B} ---")
        z = torch.randn(B, joint_dim, device=device)
        target = torch.randn(B, in_dim, device=device)

        # Forward pass
        pred = decoder(z)
        losses = ((pred - target) ** 2).mean(dim=1)

        # 1. Benchmark Python-loop fallback
        t0 = time.perf_counter()
        loop_grads = [torch.zeros_like(p) for p in params]
        loop_norms = []
        for i in range(B):
            s_grads = torch.autograd.grad(
                losses[i], params, retain_graph=(i < B - 1), allow_unused=True
            )
            norm = (sum(g.pow(2).sum() for g in s_grads if g is not None)).sqrt().item()
            norm = max(1e-8, norm)
            loop_norms.append(norm)
            coef = min(1.0, C / norm)
            for acc, g in zip(loop_grads, s_grads):
                if g is not None:
                    acc.add_(g * coef)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_loop = (time.perf_counter() - t0) * 1000.0  # ms

        # Re-compute forward for chunked
        pred = decoder(z)
        losses = ((pred - target) ** 2).mean(dim=1)

        # 2. Benchmark Chunked per-sample gradients
        t0 = time.perf_counter()
        chunk_grads = [torch.zeros_like(p) for p in params]
        chunk_norms = []
        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            c_len = end - start
            c_loss = losses[start:end]
            grad_outputs = torch.eye(c_len, device=device)
            b_grads = torch.autograd.grad(
                c_loss,
                params,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=(end < B),
                allow_unused=True,
            )
            c_sq = torch.zeros(c_len, device=device)
            for g in b_grads:
                if g is not None:
                    c_sq += g.flatten(start_dim=1).pow(2).sum(dim=1)
            c_n = torch.clamp(c_sq.sqrt(), min=1e-8)
            chunk_norms.extend(c_n.cpu().tolist())
            coef = torch.clamp(C / c_n, max=1.0)
            for i, g in enumerate(b_grads):
                if g is not None:
                    view_shape = [c_len] + [1] * (g.dim() - 1)
                    chunk_grads[i] += (g * coef.view(*view_shape)).sum(dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_chunk = (time.perf_counter() - t0) * 1000.0  # ms

        # Equality check
        max_grad_diff = max(
            torch.max(torch.abs(lg - cg)).item() for lg, cg in zip(loop_grads, chunk_grads)
        )
        max_norm_diff = max(abs(ln - cn) for ln, cn in zip(loop_norms, chunk_norms))
        speedup = t_loop / max(1e-3, t_chunk)

        print(f"  Python Loop Time : {t_loop:8.2f} ms")
        print(f"  Chunked Time     : {t_chunk:8.2f} ms")
        print(f"  Speedup Factor   : {speedup:8.2f}x faster")
        print(f"  Max Grad Diff    : {max_grad_diff:.2e} (Passed: {max_grad_diff < 1e-4})")
        print(f"  Max Norm Diff    : {max_norm_diff:.2e} (Passed: {max_norm_diff < 1e-4})")

        results.append({
            "B": B,
            "t_loop_ms": round(t_loop, 2),
            "t_chunk_ms": round(t_chunk, 2),
            "speedup": round(speedup, 2),
            "max_grad_diff": float(max_grad_diff),
            "equality_pass": bool(max_grad_diff < 1e-4),
        })

    print("\n================================================================================")
    print(" BENCHMARK SUMMARY")
    print("================================================================================")
    print(f" {'Batch Size':12s} | {'Python Loop (ms)':18s} | {'Chunked (ms)':14s} | {'Speedup':10s} | {'Equality':10s}")
    print(" " + "-" * 75)
    for r in results:
        print(f" B={r['B']:<10d} | {r['t_loop_ms']:15.2f} ms | {r['t_chunk_ms']:11.2f} ms | {r['speedup']:8.2f}x | {'PASS' if r['equality_pass'] else 'FAIL'}")
    print("================================================================================\n")

if __name__ == "__main__":
    benchmark()
