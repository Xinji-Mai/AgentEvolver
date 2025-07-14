import torch
import verl.utils.torch_functional as verl_F
from openai import OpenAI
import os
from loguru import logger
from openai import OpenAI, APIStatusError, APITimeoutError, RateLimitError
import time
import traceback
from tqdm import tqdm

__all__ = [
    "evaluate_step_flags",      # 调用 LLM 得到各 step 的 GOOD / BAD 判断
    "apply_step_mask",          # 按规则缩放 token-level advantage
]

# ————————————————————————————————————————————————————————————————
# 1. 调用评估 LLM，判定每个 step 是否 GOOD
# ————————————————————————————————————————————————————————————————
def _build_prompt(query: str,
                  rollout: str,
                  step: str,
                  overall_adv: float) -> list[dict]:
    """
    构造对话消息，要求 LLM 输出 'GOOD' 或 'BAD'（大小写皆可）。
    """
    polarity = "positive" if overall_adv > 0 else "negative"
    sys   = "You are an expert reward-model evaluator.  \nReply with **exactly one word**, either **GOOD** or **BAD** – no explanations."
    user  = (
        f"────────────────────────────────\n"
        f"USER QUERY\n{query}\n\n"
        f"ASSISTANT FULL ANSWER\n{rollout}\n\n"
        f"CURRENT ASSISTANT STEP\n{step}\n"
        f"────────────────────────────────\n\n"
        f"The total advantage (quality score) of the full answer is "
        f"**{overall_adv:+.4f}** → this is {polarity} "
        f"(positive if > 0, negative if < 0).\n\n"
        f"**Task**\n"
        f"Does the *current assistant step* improve (GOOD) or harm (BAD) "
        f"the final answer given the user query and the overall advantage?"
    )
    return [{"role": "system", "content": sys},
            {"role": "user",   "content": user}]

def _safe_query(client, model, messages, max_retries=3):
    """
    调 OpenAI SDK；若报错则抛出异常
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                timeout=30,
                max_tokens=10,  # 只需要GOOD或BAD
            )
            result = rsp.choices[0].message.content.strip()
            return result
            
        except (APIStatusError, APITimeoutError, RateLimitError) as e:
            last_exception = e
            if attempt == 0:  # 只在第一次失败时打印
                print(f"[semantic_eval] API call failed: {e.__class__.__name__}, retrying...")
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
                
        except Exception as e:
            last_exception = e
            if attempt == 0:  # 只在第一次失败时打印
                print(f"[semantic_eval] Unexpected error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0)
    
    # 如果所有重试都失败了，抛出最后一个异常
    raise last_exception

def evaluate_step_flags(tokenizer,
                        batch,
                        good_words: tuple[str, ...] = ("GOOD",),
                        bad_words:  tuple[str, ...] = ("BAD",),
                        model_name: str = "qwen-max") -> list[list[bool]]:
    """
    对于 batch 中每条样本，返回 bool 列表，长度 = step 数，
    True 表示 step 为 GOOD，False 表示 BAD。
    依赖 env_manager ➜ to_dataproto 时写入：
        • batch.non_tensor_batch['steps'] : list[list[str]]
    """
    batch_size = len(batch.batch['prompts'])
    print(f"[semantic_eval] Starting evaluation for {batch_size} samples")
    
    # 检查必要的输入
    if 'steps' not in batch.non_tensor_batch:
        raise ValueError("batch.non_tensor_batch['steps'] is required but not found")
    
    if len(batch.non_tensor_batch['steps']) != batch_size:
        raise ValueError(f"steps length ({len(batch.non_tensor_batch['steps'])}) != batch size ({batch_size})")
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    flags_per_sample: list[list[bool]] = []
    total_steps = sum(len(batch.non_tensor_batch["steps"][i]) for i in range(batch_size))
    print(f"[semantic_eval] Total steps to evaluate: {total_steps}")
    
    # 外层进度条：样本
    sample_pbar = tqdm(range(batch_size), desc="[semantic_eval] Evaluating samples", unit="sample")
    
    for idx in sample_pbar:
        try:
            query   = tokenizer.decode(batch.batch["prompts"][idx], skip_special_tokens=True)
            rollout = tokenizer.decode(batch.batch["responses"][idx], skip_special_tokens=True)
            steps   = batch.non_tensor_batch["steps"][idx]
            overall_adv = batch.batch["advantages"][idx].sum().item()

            step_flags = []
            
            # 内层进度条：当前样本的steps
            step_pbar = tqdm(steps, desc=f"  Sample {idx} steps", unit="step", leave=False)
            
            for step_text in step_pbar:
                msgs = _build_prompt(query, rollout, step_text, overall_adv)
                answer = _safe_query(client, model_name, msgs)
                
                answer_upper = answer.upper()
                is_good = answer_upper.startswith("G") or "GOOD" in answer_upper
                step_flags.append(is_good)
                
                # 更新内层进度条描述
                step_pbar.set_postfix({"result": "GOOD" if is_good else "BAD"})
            
            step_pbar.close()
            flags_per_sample.append(step_flags)
            
            # 更新外层进度条描述
            sample_pbar.set_postfix({
                "steps": len(step_flags), 
                "good": sum(step_flags), 
                "bad": len(step_flags) - sum(step_flags)
            })
            
        except Exception as e:
            sample_pbar.close()
            print(f"[semantic_eval] Error processing sample {idx}: {type(e).__name__}: {e}")
            raise
    
    sample_pbar.close()
    print(f"[semantic_eval] Evaluation completed successfully")
    return flags_per_sample


# ————————————————————————————————————————————————————————————————
# 2. 根据 GOOD/BAD 结果对 token-level advantage 做缩放
# ————————————————————————————————————————————————————————————————
def apply_step_mask(batch,
                    step_flags: list[list[bool]],
                    good_scale: float = 1.0,
                    bad_scale:  float = 0.2):
    """
    * 需要 env_manager ➜ to_dataproto 写入
        • batch.batch['step_ids'] : LongTensor (bs, resp_len)
          - 每个 response token 的 step 索引，从 0 开始；填充位置 = -1
    * 根据整体 Advantage 的正负号切换缩放系数
    """
    print(f"[apply_step_mask] Starting mask application")
    
    # 检查必要的输入
    if 'step_ids' not in batch.batch:
        raise ValueError("batch.batch['step_ids'] is required but not found")
    
    adv      = batch.batch["advantages"]          # (bs, resp_len)
    step_ids = batch.batch["step_ids"].to(adv.device)

    bs, resp_len = adv.shape
    
    if len(step_flags) != bs:
        raise ValueError(f"step_flags length ({len(step_flags)}) != batch size ({bs})")
    
    scale = torch.ones_like(adv)

    # 外层进度条：batch中的样本
    sample_pbar = tqdm(range(bs), desc="[apply_step_mask] Processing samples", unit="sample")
    
    for b in sample_pbar:
        overall_adv_sum = adv[b].sum().item()
        overall_pos = overall_adv_sum > 0      # True → overall 正
        
        # 检查当前样本的step_flags数量
        current_steps = step_flags[b]
        max_step_id = step_ids[b].max().item()
        
        # 检查step_ids的范围是否合理（忽略padding的-1）
        valid_step_ids = step_ids[b][step_ids[b] >= 0]
        if len(valid_step_ids) > 0:
            if valid_step_ids.max().item() >= len(current_steps):
                print(f"[apply_step_mask] WARNING: Sample {b} has step_id {valid_step_ids.max().item()} but only {len(current_steps)} step_flags")
                # 继续处理，但只处理有效范围内的step
        
        # 内层进度条：当前样本的step flags
        step_pbar = tqdm(enumerate(current_steps), 
                        desc=f"  Sample {b} steps", 
                        total=len(current_steps), 
                        unit="step", 
                        leave=False)
        
        good_count = 0
        bad_count = 0
        tokens_affected = 0
        
        for s, is_good in step_pbar:
            tok_mask = step_ids[b] == s            # (resp_len,)
            
            if not tok_mask.any():
                continue
                
            # 根据用户需求的4条规则：
            if overall_pos:
                # 整体advantage为正
                factor = good_scale if is_good else bad_scale  # good保持1.0，bad变0.2
            else:
                # 整体advantage为负
                factor = -bad_scale if is_good else good_scale  # good变-0.2，bad保持1.0
                
            scale[b].masked_fill_(tok_mask, factor)
            
            # 统计信息
            if is_good:
                good_count += 1
            else:
                bad_count += 1
            tokens_affected += tok_mask.sum().item()
            
            # 更新内层进度条
            step_pbar.set_postfix({
                "good": good_count,
                "bad": bad_count, 
                "tokens": tokens_affected
            })
        
        step_pbar.close()
        
        # 更新外层进度条
        sample_pbar.set_postfix({
            "adv": f"{overall_adv_sum:.2f}",
            "pos": overall_pos,
            "steps": len(current_steps),
            "tokens": tokens_affected
        })
    
    sample_pbar.close()
    
    # 保存原始advantages用于对比
    original_adv_sum = adv.sum().item()
    
    # 确保填充token（step_id == -1）保持scale=1.0不变
    print("[apply_step_mask] Applying padding mask...")
    padding_mask = (step_ids == -1)
    scale.masked_fill_(padding_mask, 1.0)
    
    batch.batch["advantages"] = adv * scale
    
    new_adv_sum = batch.batch["advantages"].sum().item()
    print(f"[apply_step_mask] Advantages sum: {original_adv_sum:.4f} -> {new_adv_sum:.4f}")
    
    # 为后续诊断留一个可视化字段
    batch.batch["semantic_scale"] = scale
    
    print("[apply_step_mask] Completed successfully")