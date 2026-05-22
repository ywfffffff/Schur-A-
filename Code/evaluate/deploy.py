"""
MoE 专家裁剪 - 第二步：部署与验证
通过 monkeypatch 方式加载 pruning_results.json，实现权重补偿与专家过滤。
"""

import torch
import torch.nn.functional as F
import json
import os
import types
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

def main():
    # ============ 配置 ============
    MODEL_PATH = "/home/lab1008/data_disk_sdc/ywf/models/Qwen/Qwen3-30B-A3B"
    RESULTS_PATH = "/home/lab1008/data_disk_sdc/ywf/data/result/pruning_results_Qwen3_no_instruct.json"
    
    print(f"1) 加载剪枝结果: {RESULTS_PATH}")
    with open(RESULTS_PATH, "r") as f:
        pruning_results = json.load(f)
        
    print(f"2) 加载模型: {MODEL_PATH} (4-bit + SDPA + Multi-Stream)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, # 显式指定精度，防止默认加载为 fp32 撑爆内存
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    def run_inference(model, tokenizer, prompt, label):
        print(f"--- {label} 推理中 ---")
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(content)
        return content

    test_prompts = [
        "If I have 3 apples and you give me 2 more, but then I eat 1, how many do I have? Explain your reasoning step by step.",
        "A shirt costs $20. There is a 20% discount today. If I buy 3 shirts, how much will I pay in total? Show the calculation.",
        "Summarize the main benefits of MoE architecture in three bullet points. Each bullet point should start with the word 'Efficiency'.",
        "Translate this into Chinese: 'The concept of Mixture-of-Experts allows large models to run efficiently by only activating a subset of parameters for each input.'"
    ]

    print(f"3) 开始对比测试...")
    results_orig = []
    for prompt in test_prompts:
        results_orig.append(run_inference(model, tokenizer, prompt, "原始模型"))

    print(f"\n4) 应用剪枝补丁...")
    patch_moe_with_pruning_results(model, pruning_results)
    
    results_pruned = []
    for prompt in test_prompts:
        results_pruned.append(run_inference(model, tokenizer, prompt, "剪枝模型"))

    print("\n" + "="*50)
    print("最终对比结果")
    print("="*50)
    for i, prompt in enumerate(test_prompts):
        print(f"\n【案例 {i+1}】: {prompt}")
        print(f"\n[原始输出]:\n{results_orig[i]}")
        print(f"\n[剪枝输出]:\n{results_pruned[i]}")
        print("-" * 30)
    
    print("\n部署验证完成。")

if __name__ == "__main__":
    main()
