import os
import json
import torch
import types
from typing import Dict, Any
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
# ================= 配置区域 =================
MODEL_PATH = "/home/lab1008/data_disk_sdc/ywf/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
RESULTS_PATH = "/home/lab1008/data_disk_sdc/ywf/data/result/pruning_results.json"
OUTPUT_FILE = "/home/lab1008/data_disk_sdc/ywf/data/Eval/Mask_Instruct.json"

# 定义所有要跑的任务及其对应的标准 few-shot 数
# 格式: { "lm_eval任务名": num_fewshot }
TASKS_CONFIG = {
    # 你指定的经典任务
    "arc_challenge": 0,   # Leaderboard标准通常为25-shot
    "arc_easy": 0,
    "boolq": 0,
    "hellaswag": 0,
    "mmlu": 0,
    "openbookqa": 0,
    "rte": 0,
    "winogrande": 0,
    # "gsm8k": 5,
    
    # # 额外推荐的任务
    # "truthfulqa_mc2": 0,   # 幻觉测试通常用0-shot
    # "ceval-valid": 5   # 中文能力测试# 代码能力快速验证
}

# ================= 补丁逻辑 =================
# def patch_moe_with_pruning_results(model, pruning_results):
#     """ (此处保留你之前的补丁逻辑代码) """
#     import torch.nn.functional as F
#     num_layers = len(model.model.layers)
#     for l in range(num_layers):
#         l_str = str(l)
#         if l_str not in pruning_results: continue
#         res = pruning_results[l_str]
#         moe_block = model.model.layers[l].mlp
#         E = int(moe_block.num_experts) if hasattr(moe_block, "num_experts") else len(moe_block.experts)
#         target_dtype = moe_block.gate.weight.dtype
#         if target_dtype == torch.uint8: target_dtype = torch.bfloat16
#         compensation = torch.zeros(E, dtype=target_dtype)
#         for idx, w in zip(res["experts"], res["weights"]):
#             compensation[idx] = w
#         moe_block._pruning_compensation = compensation
#         moe_block._retained_experts_cpu = res["experts"]
        
#         def make_new_forward(block):
#             def new_forward(self, hidden_states):
#                 logits = self.gate(hidden_states)
#                 topk_logits, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
#                 topk_w = F.softmax(topk_logits, dim=-1) if getattr(self, "norm_topk_prob", True) else topk_logits
#                 if self._pruning_compensation.device != hidden_states.device:
#                     self._pruning_compensation = self._pruning_compensation.to(hidden_states.device)
#                 comp = self._pruning_compensation[topk_idx]
#                 effective_w = topk_w * comp
#                 B, T, H = hidden_states.shape
#                 e_out = hidden_states.new_zeros((B, T, self.top_k, H))
#                 active_masks = []
#                 for e_idx in self._retained_experts_cpu:
#                     mask = (topk_idx == e_idx)
#                     if mask.any(): active_masks.append((e_idx, mask))
#                 if active_masks:
#                     streams = [torch.cuda.Stream() for _ in active_masks]
#                     for i, (e_idx, mask) in enumerate(active_masks):
#                         with torch.cuda.stream(streams[i]):
#                             pos = mask.nonzero(as_tuple=False)
#                             e_out[pos[:,0], pos[:,1], pos[:,2], :] = self.experts[e_idx](hidden_states[pos[:,0], pos[:,1], :])
#                     torch.cuda.synchronize()
#                 return (e_out * effective_w.unsqueeze(-1)).sum(dim=2)
#             return new_forward
#         moe_block.forward = types.MethodType(make_new_forward(moe_block), moe_block)


def patch_moe_with_pruning_results(model, pruning_results):
    """
    V3 优化版 (Strict V1 Logic Mode + Device Safe)
    特性：
    1. Token Permutation 加速。
    2. 保持原逻辑：Top-K 可选到被剪专家，选到后跳过计算（输出0）。
    3. 加入了设备自动对齐检查。
    """
    num_layers = len(model.model.layers)
    
    for l in range(num_layers):
        l_str = str(l)
        if l_str not in pruning_results: continue
        res = pruning_results[l_str]
        moe_block = model.model.layers[l].mlp
        
        # 1. 基础信息
        E = int(moe_block.num_experts) if hasattr(moe_block, "num_experts") else len(moe_block.experts)
        target_dtype = moe_block.gate.weight.dtype
        if target_dtype == torch.uint8: target_dtype = torch.bfloat16
        
        # 2. 垃圾桶策略映射表
        retained_list = res["experts"]
        num_retained = len(retained_list)
        TRASH_INDEX = num_retained # 垃圾桶索引
        
        # 默认指向垃圾桶
        expert_map = torch.full((E,), TRASH_INDEX, dtype=torch.long)
        for i, original_idx in enumerate(retained_list):
            expert_map[original_idx] = i 
        
        # 注册 Buffer (会自动随模型移动，但为了保险起见，forward 里也加检查)
        moe_block.register_buffer('_expert_map', expert_map)
        
        # 3. 补偿权重
        compensation = torch.zeros(E, dtype=target_dtype)
        for idx, w in zip(res["experts"], res["weights"]):
            compensation[idx] = w
        moe_block.register_buffer('_pruning_compensation', compensation)
        
        # 4. 缓存保留模块
        moe_block._retained_experts_modules = torch.nn.ModuleList([moe_block.experts[i] for i in retained_list])

        def make_new_forward(block, trash_idx):
            def new_forward(self, hidden_states):
                # --- [新增] 设备安全检查逻辑 ---
                device = hidden_states.device
                
                if self._expert_map.device != device:
                    self._expert_map = self._expert_map.to(device)
                
                if self._pruning_compensation.device != device:
                    self._pruning_compensation = self._pruning_compensation.to(device)
                    
                # 注意：此策略不需要 _expert_logit_mask，因此不进行检查
                
                # -----------------------------------
                
                B, T, H = hidden_states.shape
                
                # Part 1: Router
                logits = self.gate(hidden_states)
                # 无 Mask，允许选中被剪专家
                
                topk_logits, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
                
                if getattr(self, "norm_topk_prob", True):
                    topk_w = F.softmax(topk_logits, dim=-1)
                else:
                    topk_w = topk_logits
                
                comp = self._pruning_compensation[topk_indices]
                effective_w = topk_w * comp 
                
                # Part 2: Permutation
                x_flattened = hidden_states.unsqueeze(2).expand(-1, -1, self.top_k, -1).reshape(-1, H)
                
                expert_indices_flat = topk_indices.view(-1)
                mapped_indices = self._expert_map[expert_indices_flat] 
                
                sorted_expert_indices, sort_idx = torch.sort(mapped_indices)
                x_sorted = x_flattened[sort_idx]
                
                # Part 3: Computation
                # minlength 必须能覆盖垃圾桶索引
                counts = torch.bincount(sorted_expert_indices, minlength=trash_idx + 1)
                sections = counts.tolist()
                x_chunks = torch.split(x_sorted, sections, dim=0)
                
                # 必须初始化为 0 (因为被跳过的部分需要是 0)
                out_sorted = torch.zeros_like(x_sorted)
                
                current_offset = 0
                for i, chunk in enumerate(x_chunks):
                    chunk_len = chunk.shape[0]
                    if chunk_len > 0:
                        # 只有 i < trash_idx 才是保留专家
                        if i < trash_idx:
                            res = self._retained_experts_modules[i](chunk)
                            out_sorted[current_offset : current_offset + chunk_len] = res
                        # else: i == trash_idx -> 隐式跳过，out_sorted 对应位置保持 0
                        
                    current_offset += chunk_len
                
                # Part 4: Restore
                out_restored = torch.empty_like(out_sorted)
                out_restored.scatter_(0, sort_idx.unsqueeze(1).expand(-1, H), out_sorted)
                
                out_reshaped = out_restored.view(B, T, self.top_k, H)
                final_out = (out_reshaped * effective_w.unsqueeze(-1)).sum(dim=2)
                
                return final_out
                
            return new_forward

        moe_block.forward = types.MethodType(make_new_forward(moe_block, TRASH_INDEX), moe_block)

    return model



import types
from typing import Dict, Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def patch_moe_with_expert_masking(model, pruning_results):
    """
    Masking + Compact-ID + Single Global Permutation (方案 A)

    语义：
    - 通过 expert_mask 强制 Top-K 只能选到保留专家（被剪专家不可能被选到）
    - 通过 expert_map 将原始专家 id (0..E-1) 映射到紧凑 id (0..R-1)
    - 通过一次 sort/permutation 将所有 (token, k) 路由合并为大 batch 分桶计算
    - 输出 restore 回原顺序，并按 topk 权重加权求和

    适配：
    - Qwen3 MoE (experts: ModuleList, gate: Linear-like)
    - 单卡/多卡：buffer 运行时自动对齐到 hidden_states.device
    - 专家返回 tuple/list 时取 out[0]
    """
    num_layers = len(model.model.layers)
    print(f"Applying MASK+PERMUTE(BigBatch) Patch to {num_layers} layers...")

    for l in range(num_layers):
        l_str = str(l)
        if l_str not in pruning_results:
            continue

        res = pruning_results[l_str]
        moe_block = model.model.layers[l].mlp

        if not hasattr(moe_block, "experts"):
            continue

        experts = moe_block.experts
        E = len(experts)
        retained_list = list(res["experts"])
        R = len(retained_list)

        if R <= 0:
            raise ValueError(f"Layer {l}: retained_list is empty; cannot patch.")

        # 1) expert mask: keep=0, pruned=-1e9 (force router never select pruned experts)
        # Use float32 mask; cast to logits dtype at runtime.
        mask = torch.full((E,), -1e9, dtype=torch.float32)
        for idx in retained_list:
            mask[idx] = 0.0

        # 2) expert map: orig_id -> compact_id (0..R-1), pruned -> -1
        expert_map = torch.full((E,), -1, dtype=torch.long)
        for i, orig_idx in enumerate(retained_list):
            expert_map[orig_idx] = i

        # Register buffers (replace if exist)
        if hasattr(moe_block, "_expert_mask"):
            delattr(moe_block, "_expert_mask")
        if hasattr(moe_block, "_expert_map"):
            delattr(moe_block, "_expert_map")

        moe_block.register_buffer("_expert_mask", mask, persistent=True)
        moe_block.register_buffer("_expert_map", expert_map, persistent=True)

        # 3) compact retained experts ModuleList in compact order
        moe_block._retained_modules = torch.nn.ModuleList([experts[i] for i in retained_list])

        def make_new_forward(block, R_compact: int):
            def new_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                """
                hidden_states: [B, T, H]
                return:       [B, T, H]
                """
                device = hidden_states.device
                B, T, H = hidden_states.shape
                K = int(self.top_k)

                # --- device align buffers ---
                if self._expert_mask.device != device:
                    self._expert_mask = self._expert_mask.to(device)
                if self._expert_map.device != device:
                    self._expert_map = self._expert_map.to(device)

                # --- Router (Gate) + Masking ---
                gate_dtype = self.gate.weight.dtype
                gate_inp = hidden_states.to(gate_dtype)
                logits = self.gate(gate_inp)  # [B, T, E]

                # Apply mask in logits dtype
                masked_logits = logits + self._expert_mask.view(1, 1, -1).to(dtype=logits.dtype)

                topk_logits, topk_indices = torch.topk(masked_logits, k=K, dim=-1)  # [B,T,K], [B,T,K]

                if getattr(self, "norm_topk_prob", True):
                    topk_w = F.softmax(topk_logits, dim=-1)  # [B,T,K]
                else:
                    topk_w = topk_logits

                # --- Map to compact expert ids (0..R-1) ---
                # Since masking forbids pruned experts, mapped ids should never be -1.
                compact_idx = self._expert_map[topk_indices]  # [B,T,K]
                if torch.any(compact_idx < 0):
                    # This should not happen if mask is correct, but keep a hard guard.
                    raise RuntimeError("Compact expert mapping produced -1. Masking may be broken.")

                # --- Build flattened tokens for all K routes ---
                # x_flat: [B*T*K, H]
                x_flat = hidden_states.unsqueeze(2).expand(B, T, K, H).reshape(-1, H)

                # w_flat: [B*T*K, 1]
                w_flat = topk_w.reshape(-1, 1).to(dtype=hidden_states.dtype)

                # idx_flat: [B*T*K] compact expert ids
                idx_flat = compact_idx.reshape(-1)

                # --- Single global permutation by expert id ---
                sorted_idx, sort_perm = torch.sort(idx_flat)         # [N], [N]
                x_sorted = x_flat[sort_perm]                         # [N, H]
                w_sorted = w_flat[sort_perm]                         # [N, 1]

                # --- Split into R buckets and run experts in large batches ---
                counts = torch.bincount(sorted_idx, minlength=R_compact)  # [R]
                sections = counts.tolist()

                # Allocate output sorted buffer
                out_sorted = torch.empty_like(x_sorted)

                offset = 0
                for e_compact, n_tok in enumerate(sections):
                    if n_tok == 0:
                        continue

                    chunk = x_sorted[offset : offset + n_tok]  # [n_tok, H]
                    expert_mod = self._retained_modules[e_compact]

                    # Align expert dtype
                    try:
                        e_param = next(expert_mod.parameters())
                        chunk_inp = chunk.to(dtype=e_param.dtype)
                    except StopIteration:
                        chunk_inp = chunk  # expert has no params? unlikely, but safe.

                    out = expert_mod(chunk_inp)
                    if isinstance(out, (tuple, list)):
                        out = out[0]

                    out = out.to(dtype=out_sorted.dtype)
                    out_sorted[offset : offset + n_tok] = out

                    offset += n_tok

                # --- Restore original (token,k) order ---
                out_flat = torch.empty_like(out_sorted)  # [N, H]
                # inverse permutation via scatter
                out_flat.scatter_(0, sort_perm.unsqueeze(1).expand(-1, H), out_sorted)

                # --- Apply weights and reduce over K ---
                out_flat = out_flat * w_flat  # [N, H]
                out_btkh = out_flat.view(B, T, K, H)
                final_out = out_btkh.sum(dim=2)  # [B, T, H]

                return final_out.to(dtype=hidden_states.dtype)

            return new_forward

        moe_block.forward = types.MethodType(make_new_forward(moe_block, R), moe_block)

    return model


def pick_jsonable_eval_result(res: dict, task_name: str, shot: int) -> dict:
    # 只取最关键且可 JSON 的部分，避免 dtype / pathlib / set 等对象
    out = {
        "task": task_name,
        "num_fewshot": shot,
        "results": res.get("results", {}).get(task_name, res.get("results", None)),
        "versions": res.get("versions", None),
        "higher_is_better": res.get("higher_is_better", None),
        "n-shot": res.get("n-shot", None),
        "samples": res.get("samples", None),
    }
    return out

# ================= 主评测流程 =================
def main():
    print(f"1. 加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("2. 加载剪枝结果并打补丁...")
    with open(RESULTS_PATH, "r") as f:
        pruning_results = json.load(f)
    # patch_moe_with_pruning_results(model, pruning_results)
    patch_moe_with_expert_masking(model,pruning_results)


    print("3. 初始化 lm-eval...")
    # 强制设置 batch_size=1 以保证剪枝代码中的多流同步不冲突
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)

    print("4. 开始顺序评测所有任务...")
    final_results = {}
    
    # 获取任务列表
    tasks = list(TASKS_CONFIG.keys())
    

    # 如果 simple_evaluate 不支持动态 shot，我们可以循环调用
    # 但简单起见，这里统一使用 TASKS_CONFIG 中定义的最大公约数，或改为循环调用：
    
    all_task_results = []
    for task_name, shot in TASKS_CONFIG.items():
        print(f"\n>>> 正在评测任务: {task_name} ({shot}-shot)")
        res = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=[task_name],
            num_fewshot=shot
        )
        all_task_results.append(pick_jsonable_eval_result(res, task_name, shot))
        # ================= 核心修复：增加 default=str =================
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                # 增加 default=str，这会把所有无法序列化的对象（如 dtype）转为字符串
                json.dump(all_task_results, f, ensure_ascii=False, indent=4, default=str)
            print(f"成功保存任务 {task_name} 的结果到 {OUTPUT_FILE}")
        except Exception as e:
            print(f"保存失败，但任务已完成。错误信息: {e}")

    print("\n" + "="*50)
    print("全量评测完成！结果已保存至 full_benchmark_results.json")
    print("="*50)

if __name__ == "__main__":
    main()