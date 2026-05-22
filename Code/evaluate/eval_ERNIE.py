import os
import json
import torch
import types
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
# ================= 配置区域 =================
MODEL_PATH = "/home/lab1008/data_disk_sdc/ywf/models/baidu/ERNIE-4.5-21B-A3B-PT"
RESULTS_PATH = "/home/lab1008/data_disk_sdc/ywf/data/result/pruning_results_ERNIE.json"
OUTPUT_FILE = "/home/lab1008/data_disk_sdc/ywf/data/Eval/ERNIE_MASK.json"

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
def patch_moe_with_pruning_results(model, pruning_results):
    """
    V3 优化版 (Strict V1 Logic Mode + Device Safe)
    特性：
    1. Token Permutation 加速。
    2. 保持原逻辑：Top-K 可选到被剪专家，选到后跳过计算（输出0）。
    3. 加入了设备自动对齐检查。
    4. [New] 适配 ERNIE 模型 (expert_i 属性访问 + Tuple 返回支持)。
    5. [New] 动态查找 Gate/Router 模块名 (跳过 Dense 层)。
    """
    num_layers = len(model.model.layers)
    
    # 检测是否为 ERNIE 模型 (通过 config 或 结构特征)
    is_ernie = False
    if hasattr(model.config, "model_type") and "ernie" in model.config.model_type.lower():
        is_ernie = True
    
    # 如果 config 没写，尝试通过 layer 结构判断 (ERNIE 通常用 expert_0, expert_1...)
    if not is_ernie and num_layers > 0:
        try:
            # 检查第一层 MoE
            for l in range(num_layers):
                mlp = model.model.layers[l].mlp
                if hasattr(mlp, "expert_0"):
                    is_ernie = True
                    break
        except:
            pass
            
    print(f"Pruning Patch Mode: {'ERNIE (Tuple Return + expert_i)' if is_ernie else 'Standard (Tensor Return + experts list)'}")

    for l in range(num_layers):
        l_str = str(l)
        if l_str not in pruning_results: continue
        res = pruning_results[l_str]
        moe_block = model.model.layers[l].mlp
        
        # 1. 基础信息 & 专家获取
        if hasattr(moe_block, "num_experts"):
            E = int(moe_block.num_experts)
        elif hasattr(moe_block, "experts"):
            E = len(moe_block.experts)
        else:
            # 尝试遍历 expert_0, expert_1...
            E = 0
            while hasattr(moe_block, f"expert_{E}"):
                E += 1
        
        # [Fix] 如果没有专家 (Dense Layer)，直接跳过
        if E == 0:
            # print(f"Skipping Layer {l}: No experts found (likely a Dense layer).")
            continue

        # 动态查找 Gate 模块
        gate_name = "gate"
        gate_module = None
        # 移除 gate_proj 以避免匹配到 Dense SwiGLU 的投影层
        for name in ["gate", "router", "classifier", "w_gate"]:
            if hasattr(moe_block, name):
                gate_module = getattr(moe_block, name)
                gate_name = name
                break
        
        if gate_module is None:
            print(f"Warning: Could not find gate/router in layer {l}. Available keys: {dir(moe_block)}")
            continue

        target_dtype = gate_module.weight.dtype
        if target_dtype == torch.uint8: target_dtype = torch.bfloat16
        
        # 2. 垃圾桶策略映射表
        retained_list = res["experts"]
        num_retained = len(retained_list)
        TRASH_INDEX = num_retained # 垃圾桶索引
        
        # 默认指向垃圾桶
        expert_map = torch.full((E,), TRASH_INDEX, dtype=torch.long)
        for i, original_idx in enumerate(retained_list):
            expert_map[original_idx] = i 
        
        # 注册 Buffer
        if hasattr(moe_block, "_expert_map"):
            del moe_block._expert_map
        moe_block.register_buffer('_expert_map', expert_map)
        
        # 3. 补偿权重
        compensation = torch.zeros(E, dtype=target_dtype)
        for idx, w in zip(res["experts"], res["weights"]):
            compensation[idx] = w
        
        if hasattr(moe_block, "_pruning_compensation"):
            del moe_block._pruning_compensation
        moe_block.register_buffer('_pruning_compensation', compensation)
        
        # 4. 缓存保留模块 (兼容 list 和 expert_i)
        retained_modules = []
        for i in retained_list:
            if hasattr(moe_block, "experts") and isinstance(moe_block.experts, (list, torch.nn.ModuleList)):
                retained_modules.append(moe_block.experts[i])
            else:
                retained_modules.append(getattr(moe_block, f"expert_{i}"))
        
        moe_block._retained_experts_modules = torch.nn.ModuleList(retained_modules)

        def make_new_forward(block, trash_idx, return_tuple_flag, gate_attr_name):
            def new_forward(self, hidden_states):
                # --- [新增] 设备安全检查逻辑 ---
                device = hidden_states.device

                if self._expert_map.device != device:
                    self._expert_map = self._expert_map.to(device)
                
                if self._pruning_compensation.device != device:
                    self._pruning_compensation = self._pruning_compensation.to(device)
                    
                # -----------------------------------
                
                B, T, H = hidden_states.shape
                
                # Part 1: Router
                # 确保 gate 输入类型匹配
                gate_input = hidden_states
                gate_mod = getattr(self, gate_attr_name)
                
                if gate_mod.weight.dtype != gate_input.dtype:
                    gate_input = gate_input.to(gate_mod.weight.dtype)
                    
                logits = gate_mod(gate_input)
                
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
                
                # 必须初始化为 0
                out_sorted = torch.zeros_like(x_sorted)
                
                current_offset = 0
                for i, chunk in enumerate(x_chunks):
                    chunk_len = chunk.shape[0]
                    if chunk_len > 0:
                        # 只有 i < trash_idx 才是保留专家
                        if i < trash_idx:
                            # 确保专家输入类型匹配
                            expert_mod = self._retained_experts_modules[i]
                            # 简单 check 一下参数类型 (假设第一个参数代表整体)
                            p = next(expert_mod.parameters())
                            chunk_inp = chunk
                            if chunk_inp.dtype != p.dtype:
                                chunk_inp = chunk_inp.to(p.dtype)
                                
                            res = expert_mod(chunk_inp)
                            
                            # 转回 hidden_states 类型 (如果 expert 输出类型变了)
                            if res.dtype != hidden_states.dtype:
                                res = res.to(hidden_states.dtype)
                                
                            out_sorted[current_offset : current_offset + chunk_len] = res
                        
                    current_offset += chunk_len
                
# Part 4: Restore 路由专家
                out_restored = torch.empty_like(out_sorted)
                out_restored.scatter_(0, sort_idx.unsqueeze(1).expand(-1, H), out_sorted)
                
                out_reshaped = out_restored.view(B, T, self.top_k, H)
                # 这里 sum 的结果可能是 float32
                final_out = (out_reshaped * effective_w.unsqueeze(-1).to(out_reshaped.dtype)).sum(dim=2)
                
                # 加上 shared_experts
                shared_mod = getattr(self, "shared_experts", None)
                if shared_mod is not None:
                    # 获取 shared 权重类型并转换输入
                    s_p = next(shared_mod.parameters())
                    s_out = shared_mod(hidden_states.to(s_p.dtype))
                    if isinstance(s_out, (tuple, list)): s_out = s_out[0]
                    
                    # 累加前对齐精度
                    final_out = final_out + s_out.to(final_out.dtype)

                # 最终输出强制转回输入时的精度 (BFloat16)
                final_out = final_out.to(hidden_states.dtype)

                if return_tuple_flag:
                    return final_out, logits
                return final_out
                
            return new_forward

        moe_block.forward = types.MethodType(make_new_forward(moe_block, TRASH_INDEX, is_ernie, gate_name), moe_block)

    return model


from typing import Dict, Any, Optional




@torch.no_grad()
def patch_moe_with_expert_masking(model, pruning_results: Dict[str, Dict[str, Any]]):
    """
    ERNIE4.5 MoE 适配版：Masking + Compact-ID + Single Global Permutation (方案 A) + Shared Experts

    已按你贴的结构定制：
      - moe_block 类型：Ernie4_5_MoeSparseMoeBlock
      - 字段：moe_block.experts (ModuleList), moe_block.gate, moe_block.top_k, moe_block.shared_experts
      - 第 0 层不是 MoE：默认从 l=1 开始 patch（也会自动跳过非 MoE）
      - shared_experts：会加回共享专家输出（保持数值一致性）

    pruning_results 结构假设：
      pruning_results[layer_id_str]["experts"] = [orig_expert_idx, ...]
      (其余字段可有可无，本 patch 仅依赖 experts 列表)
    """
    # 你当前模型路径是 model.model.layers
    layers = model.model.layers
    num_layers = len(layers)
    print(f"Applying ERNIE MASK+PERMUTE(BigBatch) Patch to {num_layers} layers...")

    patched = 0

    # layer0 非 MoE：从 1 开始；同时仍做 hasattr 检查，保证鲁棒
    for l in range(1, num_layers):
        l_str = str(l)
        if l_str not in pruning_results:
            continue

        moe_block = layers[l].mlp
        if not (hasattr(moe_block, "experts") and hasattr(moe_block, "gate") and hasattr(moe_block, "top_k")):
            continue

        experts = moe_block.experts
        gate = moe_block.gate
        top_k = int(moe_block.top_k)

        # ERNIE 有 shared_experts（可能是 ModuleList / Module / None）
        shared_experts = getattr(moe_block, "shared_experts", None)

        if not isinstance(experts, torch.nn.ModuleList):
            experts = torch.nn.ModuleList(list(experts))
            moe_block.experts = experts  # 尽量回写

        E = len(experts)
        retained_list = list(pruning_results[l_str]["experts"])
        R = len(retained_list)
        if R <= 0:
            raise ValueError(f"[Layer {l}] retained_list is empty; cannot patch.")

        # --- 1) Mask: keep=0, pruned=-1e9 (force router never select pruned experts) ---
        mask = torch.full((E,), -1e9, dtype=torch.float32)
        for idx in retained_list:
            if idx < 0 or idx >= E:
                raise ValueError(f"[Layer {l}] retained expert idx {idx} out of range [0,{E}).")
            mask[idx] = 0.0

        # --- 2) Map: orig_id -> compact_id (0..R-1), pruned -> -1 ---
        expert_map = torch.full((E,), -1, dtype=torch.long)
        for i, orig_idx in enumerate(retained_list):
            expert_map[orig_idx] = i

        # Replace old buffers if any
        if hasattr(moe_block, "_expert_mask"):
            delattr(moe_block, "_expert_mask")
        if hasattr(moe_block, "_expert_map"):
            delattr(moe_block, "_expert_map")

        moe_block.register_buffer("_expert_mask", mask, persistent=True)
        moe_block.register_buffer("_expert_map", expert_map, persistent=True)

        # --- 3) Compact retained experts ---
        moe_block._retained_modules = torch.nn.ModuleList([experts[i] for i in retained_list])

        # Cache references (avoid attribute drift)
        moe_block._patched_gate = gate
        moe_block._patched_top_k = top_k
        # ERNIE 一般是 softmax topk，若你确认不是可改 False
        moe_block._patched_norm_topk_prob = True
        moe_block._patched_shared_experts = shared_experts

        def make_new_forward(R_compact: int):
            def new_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                """
                hidden_states: [B, T, H]
                return:       [B, T, H]
                """
                device = hidden_states.device
                B, T, H = hidden_states.shape
                K = int(self._patched_top_k)

                # --- device align buffers ---
                if self._expert_mask.device != device:
                    self._expert_mask = self._expert_mask.to(device)
                if self._expert_map.device != device:
                    self._expert_map = self._expert_map.to(device)

                gate_mod = self._patched_gate

                # --- Gate + Masking ---
                gate_dtype = gate_mod.weight.dtype if hasattr(gate_mod, "weight") else hidden_states.dtype
                logits = gate_mod(hidden_states.to(gate_dtype))  # [B, T, E]

                masked_logits = logits + self._expert_mask.view(1, 1, -1).to(dtype=logits.dtype)

                topk_logits, topk_indices = torch.topk(masked_logits, k=K, dim=-1)  # [B,T,K]

                if self._patched_norm_topk_prob:
                    topk_w = F.softmax(topk_logits, dim=-1)  # [B,T,K]
                else:
                    topk_w = topk_logits

                # --- Map to compact ids ---
                compact_idx = self._expert_map[topk_indices]  # [B,T,K]
                if torch.any(compact_idx < 0):
                    raise RuntimeError("Compact expert mapping produced -1. Masking may be broken.")

                # --- Flatten all (token,k) routes ---
                N = B * T * K
                x_flat = hidden_states.unsqueeze(2).expand(B, T, K, H).reshape(N, H)  # [N,H]
                w_flat = topk_w.reshape(N, 1).to(dtype=hidden_states.dtype)           # [N,1]
                idx_flat = compact_idx.reshape(N)                                     # [N]

                # --- Single global permutation by expert ---
                sorted_idx, sort_perm = torch.sort(idx_flat)  # [N], [N]
                x_sorted = x_flat[sort_perm]                  # [N,H]

                # --- Bucket & run experts in big batches ---
                counts = torch.bincount(sorted_idx, minlength=R_compact)  # [R]
                sections = counts.tolist()

                out_sorted = torch.empty_like(x_sorted)
                offset = 0
                for e_compact, n_tok in enumerate(sections):
                    if n_tok == 0:
                        continue

                    chunk = x_sorted[offset: offset + n_tok]  # [n_tok,H]
                    expert_mod = self._retained_modules[e_compact]

                    # dtype align to expert
                    try:
                        e_param = next(expert_mod.parameters())
                        chunk_inp = chunk.to(dtype=e_param.dtype)
                    except StopIteration:
                        chunk_inp = chunk

                    out = expert_mod(chunk_inp)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_sorted[offset: offset + n_tok] = out.to(dtype=out_sorted.dtype)

                    offset += n_tok

                # --- Restore order ---
                out_flat = torch.empty_like(out_sorted)
                out_flat.scatter_(0, sort_perm.unsqueeze(1).expand(-1, H), out_sorted)

                # --- Weight & reduce over K ---
                moe_out = (out_flat.to(dtype=hidden_states.dtype) * w_flat).view(B, T, K, H).sum(dim=2)

                # --- Shared experts add-back (ERNIE) ---
                shared = self._patched_shared_experts
                if shared is not None:
                    # shared_experts 可能是 ModuleList 或单个 Module
                    if isinstance(shared, torch.nn.ModuleList):
                        shared_sum = None
                        for m in shared:
                            so = m(hidden_states)
                            if isinstance(so, (tuple, list)):
                                so = so[0]
                            shared_sum = so if shared_sum is None else (shared_sum + so)
                        moe_out = moe_out + shared_sum.to(dtype=moe_out.dtype)
                    elif isinstance(shared, torch.nn.Module):
                        so = shared(hidden_states)
                        if isinstance(so, (tuple, list)):
                            so = so[0]
                        moe_out = moe_out + so.to(dtype=moe_out.dtype)
                    else:
                        # 非模块类型就忽略
                        pass

                return moe_out.to(dtype=hidden_states.dtype)

            return new_forward

        moe_block.forward = types.MethodType(make_new_forward(R), moe_block)
        patched += 1

    print(f"Patched {patched} ERNIE MoE blocks.")
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
    patch_moe_with_expert_masking(model, pruning_results)

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