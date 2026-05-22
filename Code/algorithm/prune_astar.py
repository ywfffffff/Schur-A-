import os
import json
import time
import heapq
import numpy as np

import torch
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict


@dataclass(order=True)
class PQItem:
    neg_priority: float
    payload: Any = field(compare=False)


@dataclass
class Node:
    f: float
    g: float
    S: Tuple[int, ...]
    L: Optional[torch.Tensor]
    beta: Optional[torch.Tensor]


@dataclass
class Generator:
    key: float
    parent_S: Tuple[int, ...]
    candidates: List[int]
    parent_g: float
    parent_L: Optional[torch.Tensor]
    parent_beta: Optional[torch.Tensor]


class StrictAStarPrunerSchurTorch:
    """
    Strict optimal best-first Branch&Bound for:
        maximize F(S)=u_S^T G_S^{-1} u_S, s.t. |S|=r

    Priority is an UPPER BOUND on the best completion from this state:
        key(S) = g(S) + h(S, t), t=r-|S|
    with admissible Schur-complement bound:
        u_tilde_R = u_R - G_RS beta
        Sigma_R   = G_RR - G_RS G_S^{-1} G_SR
        h(S,t)    = TopSum_t(u_tilde_i^2) / (lambda_min(Sigma_R)+eps)
    """

    def __init__(
        self,
        G: torch.Tensor,
        u: torch.Tensor,
        y2: float,
        device: str = "cpu",
        ridge: float = 1e-6,
        eps: float = 1e-12,
    ):
        self.E = int(G.shape[0])
        self.ridge = float(ridge)
        self.eps = float(eps)
        self.device = torch.device(device)

        G = G.to(self.device, dtype=torch.float64)
        u = u.to(self.device, dtype=torch.float64)

        # 1. Diagonal Preconditioning (Numerical Stability)
        G = 0.5 * (G + G.T)  # Ensure symmetry
        diag = torch.diag(G).clamp(min=1e-18)
        dinv = torch.rsqrt(diag)  # 1/sqrt(diag)
        self.dinv = dinv

        # G' = D^-1/2 G D^-1/2, u' = D^-1/2 u
        G = dinv[:, None] * G * dinv[None, :]
        u = dinv * u

        self.G = G + torch.eye(self.E, device=self.device, dtype=torch.float64) * self.ridge
        self.u = u
        self.y2 = float(y2)

    def _top_sum(self, arr_sq: torch.Tensor, t: int) -> float:
        if t <= 0 or arr_sq.numel() == 0:
            return 0.0
        t_eff = min(int(t), int(arr_sq.numel()))
        if t_eff == arr_sq.numel():
            return float(arr_sq.sum().item())
        vals, _ = torch.topk(arr_sq, t_eff, largest=True, sorted=False)
        return float(vals.sum().item())

    def schur_top_sum_ub(
        self,
        S: Tuple[int, ...],
        L: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        R: List[int],
        t: int,
    ) -> float:
        if t <= 0 or not R:
            return 0.0
        t_eff = min(int(t), len(R))

        idx_R = torch.tensor(R, device=self.device, dtype=torch.long)

        if len(S) == 0:
            u_tilde = self.u.index_select(0, idx_R)
            Sigma = self.G.index_select(0, idx_R).index_select(1, idx_R)
        else:
            idx_S = torch.tensor(S, device=self.device, dtype=torch.long)

            G_RS = self.G.index_select(0, idx_R).index_select(1, idx_S)  # m x k
            u_R = self.u.index_select(0, idx_R)
            u_tilde = u_R - (G_RS @ beta)  # m

            # W = L^{-1} G_SR, where G_SR = G_RS^T
            W = torch.linalg.solve_triangular(L, G_RS.T, upper=False)  # k x m
            Sigma = self.G.index_select(0, idx_R).index_select(1, idx_R) - (W.T @ W)  # m x m

        Sigma = 0.5 * (Sigma + Sigma.T)

        # min eigenvalue (exact) for strict version
        try:
            lam_min = torch.linalg.eigvalsh(Sigma)[0]
        except (torch._C._LinAlgError, RuntimeError):
            # Fallback: Jittering (add a small value to diagonal to improve conditioning)
            jitter = 1e-9 * torch.eye(Sigma.shape[0], device=self.device, dtype=torch.float64)
            try:
                lam_min = torch.linalg.eigvalsh(Sigma + jitter)[0]
            except (torch._C._LinAlgError, RuntimeError):
                lam_min = torch.tensor(self.eps, device=self.device, dtype=torch.float64)

        lam_min = torch.clamp(lam_min, min=self.eps)

        top_sum = self._top_sum(u_tilde * u_tilde, t_eff)
        return float(top_sum / (lam_min + self.eps))

    def incremental_add(
        self,
        S: Tuple[int, ...],
        L: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        i: int,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        c = self.G[i, i]

        if len(S) == 0:
            d = torch.sqrt(torch.clamp(c, min=1e-18))
            L_new = d.view(1, 1)
            beta_new = (self.u[i] / c).view(1)
            g_new = float((self.u[i] * beta_new[0]).item())
            return g_new, L_new, beta_new

        idx_S = torch.tensor(list(S), device=self.device, dtype=torch.long)
        v = self.G.index_select(0, idx_S)[:, i].view(-1, 1)  # k x 1
        w = torch.linalg.solve_triangular(L, v, upper=False)  # k x 1

        d_sq = c - (w * w).sum()
        d = torch.sqrt(torch.clamp(d_sq, min=1e-18))

        k = int(L.shape[0])
        L_new = torch.zeros((k + 1, k + 1), device=self.device, dtype=torch.float64)
        L_new[:k, :k] = L
        L_new[k, :k] = w.view(-1)
        L_new[k, k] = d

        u_new = torch.cat([self.u.index_select(0, idx_S), self.u[i].view(1)], dim=0).view(-1, 1)
        y = torch.linalg.solve_triangular(L_new, u_new, upper=False)
        beta_new = torch.linalg.solve_triangular(L_new.T, y, upper=True).view(-1)

        g_new = float((y * y).sum().item())
        return g_new, L_new, beta_new

    def vectorized_marginal_gains(
        self,
        S: Tuple[int, ...],
        L: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        R: List[int],
    ) -> torch.Tensor:
        if not R:
            return torch.empty((0,), device=self.device, dtype=torch.float64)

        idx_R = torch.tensor(R, device=self.device, dtype=torch.long)
        diag_R = torch.diag(self.G).index_select(0, idx_R)

        if len(S) == 0:
            u_R = self.u.index_select(0, idx_R)
            return (u_R * u_R) / torch.clamp(diag_R, min=1e-18)

        idx_S = torch.tensor(list(S), device=self.device, dtype=torch.long)
        A = self.G.index_select(0, idx_S).index_select(1, idx_R)  # k x m
        W = torch.linalg.solve_triangular(L, A, upper=False)       # k x m
        w_norm2 = (W * W).sum(dim=0)                               # m

        den = torch.clamp(diag_R - w_norm2, min=1e-18)
        u_R = self.u.index_select(0, idx_R)
        num = u_R - (A.T @ beta)
        return (num * num) / den

    def solve(
        self,
        r: int,
        base_ids: Optional[List[int]] = None,
        greedy_order: Optional[List[int]] = None,
        support: Optional[np.ndarray] = None,
        support_threshold: int = 64,
    ) -> Tuple[float, List[int], torch.Tensor, Dict[str, float]]:
        r = int(r)
        assert 0 <= r <= self.E

        pq: List[PQItem] = []
        closed: set = set()

        best_reward = -float("inf")
        best_S: Tuple[int, ...] = ()
        best_beta = torch.zeros((0,), device=self.device, dtype=torch.float64)

        # 1. Pre-selection (Start from base_ids)
        start_S = ()
        start_g = 0.0
        start_L = None
        start_beta = None

        if base_ids is not None and len(base_ids) > 0:
            # Fix base_ids as the starting point
            for i in base_ids:
                start_g, start_L, start_beta = self.incremental_add(start_S, start_L, start_beta, i)
                start_S = tuple(sorted(start_S + (i,)))
            print(f"  [PreSelect] Starting from {len(start_S)} experts, g={start_g:.6e}")

        # 2. Greedy Warm Start (Feasible solution check)
        if greedy_order is not None and len(greedy_order) >= r:
            greedy_S_set = set(greedy_order[:r])
            # A solution is feasible in the restricted space ONLY if:
            # 1. It contains all base_ids
            # 2. It does NOT contain any experts that would be filtered by support
            is_feasible = True
            if base_ids:
                for b in base_ids:
                    if b not in greedy_S_set:
                        is_feasible = False
                        break
            
            if is_feasible and support is not None:
                # Check if any expert in greedy_S_set has support < support_threshold
                # BUT: base_ids are always allowed even if low support
                base_ids_set = set(base_ids) if base_ids else set()
                for i in greedy_order[:r]:
                    if i not in base_ids_set and support[i] < support_threshold:
                        is_feasible = False
                        break
            
            if is_feasible:
                # Calculate reward for greedy solution
                g_greedy = 0.0
                L_greedy = None
                beta_greedy = None
                curr_S = ()
                for i in greedy_order[:r]:
                    g_greedy, L_greedy, beta_greedy = self.incremental_add(curr_S, L_greedy, beta_greedy, i)
                    curr_S = tuple(sorted(curr_S + (i,)))
                
                best_reward = g_greedy
                best_S = tuple(sorted(greedy_order[:r]))
                best_beta = beta_greedy
                print(f"  [WarmStart] Feasible greedy reward: {best_reward:.6e}")
            else:
                print(f"  [WarmStart] Greedy solution is INFEASIBLE (missing base_ids or filtered by support), skipping.")

        # Root generator: candidates are all experts not in start_S
        chosen = set(start_S)
        
        # 3. Support-based Filtering
        if support is not None:
            low_support_mask = (support < support_threshold)
            # Ensure we don't filter out experts that are already in start_S
            for i in start_S:
                low_support_mask[i] = False
            
            n_filtered = int(np.sum(low_support_mask))
            if n_filtered > 0:
                print(f"  [SupportFilter] Filtering out {n_filtered} experts with support < {support_threshold}")
            
            all_R = [j for j in range(self.E) if j not in chosen and not low_support_mask[j]]
        else:
            all_R = [j for j in range(self.E) if j not in chosen]
        
        t_need = r - len(start_S)
        initial_ub = best_reward  # 先设为 warm start 的值（如果有的话）
        if t_need > 0:
            if len(all_R) < t_need:
                print(f"  [Warning] Not enough candidates ({len(all_R)}) to reach target r ({r}) after filtering. Using all available.")
                all_R = [j for j in range(self.E) if j not in chosen]

            root_h = self.schur_top_sum_ub(start_S, start_L, start_beta, all_R, t_need)
            root_key = start_g + root_h
            initial_ub = max(initial_ub, root_key)  # 取 max(greedy, root_f)
            if root_key > best_reward:
                heapq.heappush(
                    pq,
                    PQItem(
                        -root_key,
                        Generator(
                            key=root_key,
                            parent_S=start_S,
                            candidates=all_R,
                            parent_g=start_g,
                            parent_L=start_L,
                            parent_beta=start_beta,
                        ),
                    ),
                )
        elif t_need == 0:
            # Already at target r
            initial_ub = start_g
            if start_g > best_reward:
                best_reward = start_g
                best_S = start_S
                best_beta = start_beta

        n_pop = n_node = n_gen = n_prune = n_push = 0
        t_start = time.time()

        while pq:
            item = heapq.heappop(pq)
            n_pop += 1
            ub = -item.neg_priority
            obj = item.payload

            if ub <= best_reward:
                break

            if isinstance(obj, Generator):
                n_gen += 1
                gen = obj

                if gen.key <= best_reward:
                    n_prune += 1
                    continue
                if not gen.candidates:
                    continue

                i = gen.candidates[0]
                remaining = gen.candidates[1:]

                new_S = tuple(sorted(gen.parent_S + (i,)))
                if new_S not in closed:
                    g_new, L_new, beta_new = self.incremental_add(gen.parent_S, gen.parent_L, gen.parent_beta, i)

                    t_rem = r - len(new_S)
                    # 子节点剩余集合直接用 remaining（它就是 R \ {i}）
                    h_ub = self.schur_top_sum_ub(new_S, L_new, beta_new, remaining, t_rem)
                    f_new = g_new + h_ub

                    if f_new > best_reward:
                        heapq.heappush(pq, PQItem(-f_new, Node(f=f_new, g=g_new, S=new_S, L=L_new, beta=beta_new)))
                        n_push += 1
                    else:
                        n_prune += 1

                if remaining:
                    # 父节点的“继续生成器”：key = g(parent)+h(parent, t_need)，R 用 remaining
                    t_need = r - len(gen.parent_S)
                    h_parent = self.schur_top_sum_ub(gen.parent_S, gen.parent_L, gen.parent_beta, remaining, t_need)
                    new_key = gen.parent_g + h_parent

                    if new_key > best_reward:
                        heapq.heappush(
                            pq,
                            PQItem(
                                -new_key,
                                Generator(
                                    key=new_key,
                                    parent_S=gen.parent_S,
                                    candidates=remaining,
                                    parent_g=gen.parent_g,
                                    parent_L=gen.parent_L,
                                    parent_beta=gen.parent_beta,
                                ),
                            ),
                        )
                        n_push += 1
                    else:
                        n_prune += 1
                continue

            n_node += 1
            node: Node = obj

            if node.f <= best_reward:
                n_prune += 1
                continue

            if node.S in closed:
                continue
            closed.add(node.S)

            if len(node.S) == r:
                if node.g > best_reward:
                    best_reward = node.g
                    best_S = node.S
                    best_beta = node.beta
                continue

            # Expand: order candidates by true marginal gain (for search efficiency only)
            # Note: candidates are already filtered in the Generator
            R = obj.candidates if isinstance(obj, Generator) else [j for j in range(self.E) if j not in set(node.S)]
            
            # For simplicity and correctness, let's re-filter R here if support is provided.
            chosen = set(node.S)
            if support is not None:
                low_support_mask = (support < support_threshold)
                for i in node.S:
                    low_support_mask[i] = False
                R = [j for j in range(self.E) if j not in chosen and not low_support_mask[j]]
            else:
                R = [j for j in range(self.E) if j not in chosen]

            if not R:
                continue

            gains = self.vectorized_marginal_gains(node.S, node.L, node.beta, R)
            order = torch.argsort(-gains)
            sorted_R = [R[idx] for idx in order.tolist()]

            t_need = r - len(node.S)
            h_ub = self.schur_top_sum_ub(node.S, node.L, node.beta, sorted_R, t_need)
            gen_key = node.g + h_ub

            if gen_key <= best_reward:
                n_prune += 1
                continue

            heapq.heappush(
                pq,
                PQItem(
                    -gen_key,
                    Generator(
                        key=gen_key,
                        parent_S=node.S,
                        candidates=sorted_R,
                        parent_g=node.g,
                        parent_L=node.L,
                        parent_beta=node.beta,
                    ),
                ),
            )
            n_push += 1

        t_end = time.time()
        
        # Restore beta to original coordinates: beta_orig = D^-1/2 * beta_scaled
        if len(best_S) > 0:
            idx_best = torch.tensor(list(best_S), device=self.device, dtype=torch.long)
            best_beta = self.dinv[idx_best] * best_beta

        # ========== Retrospective Path Analysis ==========
        # 沿最优路径回溯，计算每步的 h 紧度
        path_tightness = []
        if len(best_S) > 0:
            path_S = ()
            path_L = None
            path_beta = None
            path_g = 0.0
            
            remaining = list(best_S)
            for step in range(len(best_S)):
                R_now = [j for j in remaining if j not in set(path_S)]
                t_rem = r - len(path_S)
                
                if t_rem > 0 and R_now:
                    h_now = self.schur_top_sum_ub(path_S, path_L, path_beta, R_now, t_rem)
                    actual_remaining = best_reward - path_g
                    tightness = actual_remaining / h_now if h_now > 1e-12 else 1.0
                    
                    path_tightness.append({
                        "step": step,
                        "|S|": len(path_S),
                        "h": float(h_now),
                        "actual_remaining": float(actual_remaining),
                        "tightness": float(tightness),
                    })
                
                if not R_now:
                    break
                gains = self.vectorized_marginal_gains(path_S, path_L, path_beta, R_now)
                best_idx = int(torch.argmax(gains).item())
                next_i = R_now[best_idx]
                
                path_g, path_L, path_beta = self.incremental_add(path_S, path_L, path_beta, next_i)
                path_S = tuple(sorted(path_S + (next_i,)))
                remaining = [j for j in remaining if j != next_i]

        avg_tightness = np.mean([p["tightness"] for p in path_tightness]) if path_tightness else 1.0
        min_tightness = min([p["tightness"] for p in path_tightness]) if path_tightness else 1.0

        stats = {
            "time_sec": float(t_end - t_start),
            "popped": float(n_pop),
            "pushed": float(n_push),
            "node_popped": float(n_node),
            "gen_popped": float(n_gen),
            "pruned": float(n_prune),
            "closed_size": float(len(closed)),
            "initial_ub": float(initial_ub),
            "final_reward": float(best_reward),
            "ub_gap": float(initial_ub - best_reward),
            "ub_tightness": float(best_reward / initial_ub) if initial_ub > 0 else 1.0,
            "avg_path_tightness": float(avg_tightness),
            "min_path_tightness": float(min_tightness),
            "path_tightness_detail": path_tightness,
        }
        return best_reward, list(best_S), best_beta, stats


def process_layer(args_tuple):
    l, G_np, u_np, y2, r, base_ids, greedy_order, support, support_threshold, device, ridge, eps = args_tuple
    G_t = torch.from_numpy(G_np)
    u_t = torch.from_numpy(u_np)
    pruner = StrictAStarPrunerSchurTorch(G_t, u_t, y2, device=device, ridge=ridge, eps=eps)
    reward, best_S, best_beta, st = pruner.solve(r, base_ids=base_ids, greedy_order=greedy_order, support=support, support_threshold=support_threshold)
    ratio = (reward / y2) if y2 > 0 else 0.0
    
    print(f"Layer {l:02d} | r={r:3d} | reward={reward:.6e} | ratio={ratio:.6f} | time={st['time_sec']:.3f}s | popped={int(st['popped'])}")
    
    return str(l), {
        "r": int(r),
        "experts": best_S,
        "weights": best_beta.detach().cpu().tolist(),
        "reward": float(reward),
        "ratio": float(ratio),
        "stats": st,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strict optimal MoE expert pruning (Schur-complement UB, Torch)")
    parser.add_argument("--data_dir", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/stats_4_ERNIE_1")
    parser.add_argument("--output_file", type=str, default="/home/lab1008/data_disk_sdc/ywf/data/result")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=28)
    parser.add_argument("--support_threshold", type=int, default=64)
    parser.add_argument(
    "--keep_ratio",
    type=float,
    default=None,
    help="If set (0<keep_ratio<=1), ignore r_per_layer.json 'r' and keep fixed ratio of experts per layer.",
)

    args = parser.parse_args()

    G_all = np.load(os.path.join(args.data_dir, "G.npy"))
    u_all = np.load(os.path.join(args.data_dir, "u.npy"))
    y2_all = np.load(os.path.join(args.data_dir, "y2.npy"))
    
    support_path = os.path.join(args.data_dir, "support.npy")
    if os.path.exists(support_path):
        support_all = np.load(support_path)
    else:
        support_all = None
        print(f"Warning: {support_path} not found. Support-based filtering disabled.")

    with open(os.path.join(args.data_dir, "r_per_layer.json"), "r") as f:
        r_info = json.load(f)
    
    SKIP_LAYERS = {0, 1, 27}

    L_layers = int(G_all.shape[0])
    tasks = []
    results = {}
    for l in range(L_layers):
        l_str = str(l)
        if l_str not in r_info["r_per_layer"]:
            continue
            # === 新增：跳过层直接保留全部专家，weights=1 ===
        if l in SKIP_LAYERS:
            E = int(G_all.shape[2])  # G_all shape: [L, E, E]
            experts = list(range(E))
            weights = [1.0] * E
            results[l_str] = {
                "r": E,                         # 这里设为 E（全保留）。如果你更想保留原 r，也可以改为 int(r_info["r_per_layer"][l_str]["r"])
                "experts": experts,
                "weights": weights,
                "reward": float("nan"),         # 不搜索就没有严格 reward，填 NaN/0 都行
                "ratio": float("nan"),
                "stats": {"skipped": True},
            }
            print(f"Layer {l:02d} | SKIP search | keep all experts E={E} | weights=1")
            continue
        layer_info = r_info["r_per_layer"][l_str]
        if args.keep_ratio is not None:
            kr = float(args.keep_ratio)
            if not (0.0 < kr <= 1.0):
                raise ValueError(f"--keep_ratio must be in (0,1], got {kr}")
            r = int(round(E * kr))
            r = max(1, min(E, r))  # safety clamp
        else:
            r = int(layer_info["r"])
        base_ids = layer_info.get("base_ids", [])
        greedy_order = layer_info.get("selected_order", [])
        if args.keep_ratio is not None:
            # greedy_order 可能不足以 warm start
            if not greedy_order or len(greedy_order) < r:
                greedy_order = None
        support = support_all[l] if support_all is not None else None
        
        tasks.append((l, G_all[l], u_all[l], float(y2_all[l]), r, base_ids, greedy_order, support, args.support_threshold, args.device, args.ridge, args.eps))

    print(f"Starting pruning for {len(tasks)} layers on {args.device} with {args.num_workers} workers...")
    
    t0 = time.time()

    if args.num_workers > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            results_list = pool.map(process_layer, tasks)
        results.update(dict(results_list))
    else:
        for task in tasks:
            k, v = process_layer(task)
            results[k] = v

    t1 = time.time()
    total_reward = sum(v["reward"] for v in results.values())
    total_y2 = float(np.sum(y2_all))
    print(f"All layers done in {t1 - t0:.2f}s | Total EV ratio = {total_reward / total_y2:.6f}")

    os.makedirs(args.output_file, exist_ok=True)
    out_path = os.path.join(args.output_file, "pruning_results_ERNIE.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
