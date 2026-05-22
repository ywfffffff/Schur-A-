"""
MoE 专家裁剪 - 第一步：统计量收集
使用 forward monkeypatch 方式精确收集 y2, u, G 统计量（C_i = topk_softmax_weight * expert_output）
- 不修改 transformers 源码文件
- 修复 device_map="auto" 的输入设备问题
- 将统计量累加从 O(E^2 * B*T*H) 改为 O(B*T*k^2*H)
- 新增：记录总 token 数 N，并在保存前对 y2/u/G 做 1/N 归一化（不改变 argmax，仅缩放 reward）
"""

import types
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import os
import json

# ============ 配置 ============
MODEL_PATH = "/home/lab1008/data_disk_sdc/ywf/models/Qwen/Qwen3-30B-A3B"
CALIBRATION_DATA = "/home/lab1008/data_disk_sdc/ywf/data/Calibrate/qwen_math_128.pt"
OUTPUT_DIR = "/home/lab1008/data_disk_sdc/ywf/data/stats_math"
BATCH_SIZE = 1  # 显存紧张建议 1

# 是否在保存前做 1/N 归一化（推荐 True）
NORMALIZE_BY_TOKENS = True

# ============ Forward 补丁 ============
def patch_moe_for_stats(moe_block):
    """
    通用 MoE 补丁函数，支持 Qwen3 和 ERNIE-4.5。
    """
    orig_forward = moe_block.forward
    
    # 兼容性获取专家数和 top_k
    # Qwen3: num_experts / top_k
    # DeepSeek-V2: n_routed_experts / num_experts_per_tok
    E = int(getattr(moe_block, "num_experts",
            getattr(moe_block, "n_routed_experts", 0)))
    if E == 0 and hasattr(moe_block, "experts"):
        E = len(moe_block.experts)
    
    k = int(getattr(moe_block, "top_k",
            getattr(moe_block, "num_experts_per_tok", 0)))
    
    def forward_with_cache(self, hidden_states):
        # 1. 调用原始 forward 获取真实的 y (包含 shared experts 等)
        # 这样可以确保 y 是完整的输出，统计量收集更准确
        y_true = orig_forward(hidden_states)
        
        # 处理元组返回的情况 (例如 ERNIE-4.5 可能返回 (output, router_logits))
        y_tensor = y_true[0] if isinstance(y_true, tuple) else y_true
        
        # [NEW] 如果存在 shared_expert / shared_experts，从 y 中扣除其贡献
        # Qwen3: 无 shared expert
        # ERNIE-4.5: shared_expert (单数)
        # DeepSeek-V2: shared_experts (复数, 单个 Module)
        shared_mod = getattr(self, "shared_expert", None) or getattr(self, "shared_experts", None)
        if shared_mod is not None:
            # 确保 shared_expert 输入设备匹配
            try:
                shared_param = next(shared_mod.parameters())
                shared_inp = hidden_states
                if shared_inp.device != shared_param.device or shared_inp.dtype != shared_param.dtype:
                    shared_inp = shared_inp.to(device=shared_param.device, dtype=shared_param.dtype)
                
                shared_out = shared_mod(shared_inp)
                if isinstance(shared_out, (tuple, list)):
                    shared_out = shared_out[0]
                
                # 确保输出回到 y_tensor 的设备和精度
                if shared_out.device != y_tensor.device or shared_out.dtype != y_tensor.dtype:
                    shared_out = shared_out.to(device=y_tensor.device, dtype=y_tensor.dtype)
                
                y_tensor = y_tensor - shared_out
            except StopIteration:
                pass  # 没有参数的模块，跳过

        x = hidden_states
        B, T, H = x.shape

        # 确保 gate 的输入设备和精度匹配
        gate_param = next(self.gate.parameters())
        gate_inp = x
        if gate_inp.device != gate_param.device or gate_inp.dtype != gate_param.dtype:
            gate_inp = gate_inp.to(device=gate_param.device, dtype=gate_param.dtype)
            
        gate_output = self.gate(gate_inp)
        
        # DeepSeek MoEGate 直接返回 (topk_idx, topk_weight) 元组
        # Qwen3 MoEGate 返回 logits 张量，需要再做 topk
        if isinstance(gate_output, tuple):
            # DeepSeek 模式：gate 直接输出 (topk_idx, topk_w)
            # 注意：DeepSeek gate 处理的是 flatten 后的 [B*T, k]，需要 reshape 回 [B, T, k]
            topk_idx, topk_w = gate_output[0], gate_output[1]
            if topk_idx.device != x.device:
                topk_idx = topk_idx.to(x.device)
            if topk_w.device != x.device:
                topk_w = topk_w.to(x.device)
            # reshape: [B*T, k] -> [B, T, k]
            topk_idx = topk_idx.view(B, T, -1)
            topk_w = topk_w.view(B, T, -1)
        else:
            # Qwen3 模式：gate 返回 logits [B,T,E]
            logits = gate_output
            if logits.device != x.device:
                logits = logits.to(x.device)
                
            topk_logits, topk_idx = torch.topk(logits, k=k, dim=-1)  # [B,T,k]

            _norm = getattr(self, "norm_topk_prob",
                    getattr(getattr(self, "config", None), "norm_topk_prob", None))
            if _norm is None:
                scoring = getattr(self, "scoring_func", "softmax")
                _norm = (scoring == "softmax")
            if _norm:
                topk_w = F.softmax(topk_logits, dim=-1)
            else:
                topk_w = topk_logits.float().softmax(dim=-1)


        e_out = x.new_zeros((B, T, k, H))

        for e in range(E):
            mask = (topk_idx == e)  # [B,T,k]
            if not mask.any():
                continue
            
            # 兼容性获取专家模块
            if hasattr(self, "experts") and isinstance(self.experts, (list, torch.nn.ModuleList)):
                expert_module = self.experts[e]
            else:
                expert_module = getattr(self, f"expert_{e}", None)
            
            if expert_module is None:
                raise AttributeError(f"无法找到专家 {e} 的实现")

            pos = mask.nonzero(as_tuple=False)      # [M,3] (b,t,r)
            inp = x[pos[:, 0], pos[:, 1], :]        # [M,H]
            
            # 多卡与多精度支持：确保输入在专家所在的设备和精度上
            expert_param = next(expert_module.parameters())
            expert_device = expert_param.device
            expert_dtype = expert_param.dtype
            
            curr_inp = inp
            if curr_inp.device != expert_device or curr_inp.dtype != expert_dtype:
                curr_inp = curr_inp.to(device=expert_device, dtype=expert_dtype)
            
            out = expert_module(curr_inp)                # [M,H]
            
            # 确保输出回到原始设备和精度
            if out.device != x.device or out.dtype != x.dtype:
                out = out.to(device=x.device, dtype=x.dtype)
                
            e_out[pos[:, 0], pos[:, 1], pos[:, 2], :] = out

        self._stats_cache = {
            "topk_idx": topk_idx.detach(),
            "topk_w": topk_w.detach(),
            "e_out": e_out.detach(),
            "y": y_tensor.detach(),
        }
        return y_true

    moe_block.forward = types.MethodType(forward_with_cache, moe_block)
    return orig_forward


# ============ 统计量累加器 ============
class MoEStatsAccumulator:
    def __init__(self, num_layers, num_experts, top_k):
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)

        self.y2 = np.zeros(self.num_layers, dtype=np.float64)
        self.u = np.zeros((self.num_layers, self.num_experts), dtype=np.float64)
        self.G = np.zeros((self.num_layers, self.num_experts, self.num_experts), dtype=np.float64)
        self.support = np.zeros((self.num_layers, self.num_experts), dtype=np.int64)

        # 新增：token 计数（每层相同，但用 per-layer 保存更稳妥）
        self.token_count = np.zeros(self.num_layers, dtype=np.int64)

    def accumulate_layer(self, layer_idx: int, cache: dict):
        E = self.num_experts
        k = self.top_k

        topk_idx = cache["topk_idx"].to(torch.int64)  # [B,T,k]
        topk_w = cache["topk_w"].float()              # [B,T,k]
        e_out = cache["e_out"].float()                # [B,T,k,H]
        y = cache["y"].float()                        # [B,T,H]

        B, T, k2, H = e_out.shape
        assert k2 == k, f"cache k={k2} != accumulator k={k}"

        N = B * T
        self.token_count[layer_idx] += int(N)

        idx = topk_idx.reshape(N, k).cpu().numpy()    # [N,k] int
        w = topk_w.reshape(N, k)                      # [N,k] torch
        o = e_out.reshape(N, k, H)                    # [N,k,H] torch
        yy = y.reshape(N, H)                          # [N,H] torch

        self.y2[layer_idx] += float((yy * yy).sum().item())

        c = o * w.unsqueeze(-1)  # [N,k,H]

        dots = (yy.unsqueeze(1) * c).sum(dim=-1).cpu().numpy()  # [N,k]
        for r in range(k):
            e_ids = idx[:, r]
            self.u[layer_idx] += np.bincount(e_ids, weights=dots[:, r], minlength=E)
            self.support[layer_idx] += np.bincount(e_ids, minlength=E).astype(np.int64)

        gram = torch.einsum("nkh,nlh->nkl", c, c).cpu().numpy()  # [N,k,k]

        ea = idx[:, :, None]                   # [N,k,1]
        eb = idx[:, None, :]                   # [N,1,k]
        pair_ids = (ea * E + eb).reshape(-1)   # [N*k*k]
        weights = gram.reshape(-1).astype(np.float64)

        G_flat = np.bincount(pair_ids, weights=weights, minlength=E * E).astype(np.float64)
        self.G[layer_idx] += G_flat.reshape(E, E)

    def normalize_by_tokens(self):
        # 取每层 token_count；若有层没统计到（理论不该发生），跳过
        for l in range(self.num_layers):
            N = int(self.token_count[l])
            if N <= 0:
                continue
            invN = 1.0 / float(N)
            self.y2[l] *= invN
            self.u[l, :] *= invN
            self.G[l, :, :] *= invN

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "y2.npy"), self.y2)
        np.save(os.path.join(output_dir, "u.npy"), self.u)
        np.save(os.path.join(output_dir, "G.npy"), self.G)
        np.save(os.path.join(output_dir, "support.npy"), self.support)
        np.save(os.path.join(output_dir, "token_count.npy"), self.token_count)

        meta = {
            "num_layers": int(self.num_layers),
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
            "Ci_definition": "Ci(token,rank) = softmax_topk_weight(token,rank) * expert_output(token,rank)",
            "note": "forward is monkeypatched in-process; no transformers source files modified",
            "normalized_by_tokens": bool(NORMALIZE_BY_TOKENS),
            "token_count_per_layer": [int(x) for x in self.token_count.tolist()],
        }
        with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"统计量已保存到: {output_dir}")
        print(f"  - y2: {self.y2.shape}")
        print(f"  - u:  {self.u.shape}")
        print(f"  - G:  {self.G.shape}")
        print(f"  - support: {self.support.shape}")
        print(f"  - token_count: {self.token_count.shape} (per-layer)")


# ============ 通用 MoE 块探测 ============
# 不同模型将 MoE 子模块挂在不同属性名下：
#   Qwen3 / DeepSeek-V2：layer.mlp
#   Mixtral-8x7B：       layer.block_sparse_moe
#   部分自定义模型：       layer.moe
MOE_ATTR_CANDIDATES = ["block_sparse_moe", "mlp", "moe"]

def get_moe_block(layer):
    """从层中提取 MoE 子模块，返回 (attr_name, module) 或 (None, None)。"""
    for attr in MOE_ATTR_CANDIDATES:
        block = getattr(layer, attr, None)
        if block is not None and (
            hasattr(block, "num_experts")
            or hasattr(block, "n_routed_experts")
            or hasattr(block, "experts")
        ):
            return attr, block
    return None, None


def main():
    # 动态识别模型名称
    model_name = os.path.basename(MODEL_PATH.rstrip("/"))
    print("=" * 60)
    print(f"MoE 统计量收集 - {model_name} (通用 Patch 版)")
    print("=" * 60)

    if not os.path.exists(CALIBRATION_DATA):
        print(f"错误：找不到 {CALIBRATION_DATA}")
        return

    print(f"\n1) 加载校准数据: {CALIBRATION_DATA}")
    input_ids = torch.load(CALIBRATION_DATA)
    if isinstance(input_ids, list):
        print(f"   形状: 包含 {len(input_ids)} 条数据的列表 (长度不一)")
    else:
        print(f"   形状: {tuple(input_ids.shape)}")

    print(f"\n2) 加载模型: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 打印第 0 层结构，方便确认 MoE 子模块属性名
    print("\n--- 模型架构（第 0 层）---")
    print(model.model.layers[0])
    print("--- 模型架构结束 ---\n")

    num_layers = len(model.model.layers)
    
    # 动态寻找第一个 MoE 层来获取专家数和 top_k
    moe_example = None
    moe_attr_name = None
    for layer in model.model.layers:
        attr, block = get_moe_block(layer)
        if block is not None:
            moe_example = block
            moe_attr_name = attr
            break
    
    if moe_example is None:
        print(f"错误：在模型中找不到任何 MoE 层！（候选属性：{MOE_ATTR_CANDIDATES}）")
        return

    num_experts = int(getattr(moe_example, "num_experts",
                     getattr(moe_example, "n_routed_experts", 0)))
    if num_experts == 0 and hasattr(moe_example, "experts"):
        num_experts = len(moe_example.experts)
    top_k = int(getattr(moe_example, "top_k",
                getattr(moe_example, "num_experts_per_tok", 0)))
    
    print(f"   MoE 子模块属性名: '{moe_attr_name}'")
    print(f"   层数: {num_layers}, 专家数: {num_experts}, top_k: {top_k}, norm_topk_prob: {getattr(moe_example, 'norm_topk_prob', 'N/A')}")

    print(f"\n3) 为所有 MoE 层打 forward 补丁（仅当前进程内生效）")
    patched_count = 0
    for layer in model.model.layers:
        _, block = get_moe_block(layer)
        if block is not None:
            patch_moe_for_stats(block)
            patched_count += 1
    print(f"   已打补丁 {patched_count} 层 (跳过了非 MoE 层)")

    accumulator = MoEStatsAccumulator(num_layers=num_layers, num_experts=num_experts, top_k=top_k)

    num_samples = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[0]
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n4) 开始收集统计量：batch_size={BATCH_SIZE}, total_batches={num_batches}")

    input_device = model.model.embed_tokens.weight.device
    did_check = True

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="处理中"):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, num_samples)

            if isinstance(input_ids, list):
                batch_list = input_ids[start:end]
                max_len = max(x.shape[1] for x in batch_list)
                padded_batch = []
                for x in batch_list:
                    if x.shape[1] < max_len:
                        import torch.nn.functional as F
                        padded_batch.append(F.pad(x, (0, max_len - x.shape[1]), value=0))
                    else:
                        padded_batch.append(x)
                batch_input = torch.cat(padded_batch, dim=0).to(input_device)
            else:
                batch_input = input_ids[start:end].to(input_device)

            _ = model(batch_input, use_cache=False)

            for layer_idx, layer in enumerate(model.model.layers):
                _, block = get_moe_block(layer)
                cache = getattr(block, "_stats_cache", None) if block is not None else None
                if cache is None:
                    continue

                if not did_check and layer_idx == 0:
                    y = cache["y"].float()
                    recon = (cache["e_out"].float() * cache["topk_w"].float().unsqueeze(-1)).sum(dim=2)
                    max_err = (y - recon).abs().max().item()
                    print(f"\n[Sanity] layer0 max|y - sum(w*e_out)| = {max_err:.6e}")
                    did_check = True

                accumulator.accumulate_layer(layer_idx, cache)
                block._stats_cache = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if NORMALIZE_BY_TOKENS:
        print("\n5) 统计量归一化：对每层 y2/u/G 除以该层 token_count（不改变最优解，仅缩放 reward）")
        accumulator.normalize_by_tokens()

    print(f"\n6) 保存统计量到 {OUTPUT_DIR}")
    accumulator.save(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
