import sys
import os
from typing import Any, Callable, Dict, List
from copy import deepcopy
import torch
from verl.utils.model import compute_position_id_with_mask

from recipe.beyond_agent.schema import Experience

class NaiveDataFlow:
    def __init__(self, query: Experience, llm_chat_fn: callable, sampling_params: Dict, tokenizer: Any):
        self.query = query
        self.llm_chat_fn = llm_chat_fn
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer

    def execute(self):
        # a simple debug dataflow
        traj_messages, rewards_dict = self._run_agent(self.query.messages)
        # produce a single one-turn experience
        out_experience = self._make_experience(traj_messages, rewards_dict, experience=self.query)

        return out_experience
    
    def _run_agent(self, messages, **kwargs):
        output_msg = self.llm_chat_fn(messages)
        return messages + output_msg, {"outcome": 0.0, "description": "Outcome 1 denotes success, and 0 denotes failure."}
    
    def _make_experience(self, messages: List, rewards_dict: Dict, experience: Experience):
        out_experience = deepcopy(experience)
        out_experience.messages = messages

        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        outputs = self.tokenizer(full_text, return_tensors="pt", padding=False)
        input_ids = outputs["input_ids"][0].tolist()  # 移除batch维度
        attention_mask = outputs["attention_mask"][0].tolist()
        
        # 假设最后一条message是assistant的回复
        prompt_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_outputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False)
        prompt_ids = prompt_outputs["input_ids"][0].tolist()
        prompt_attention_mask = prompt_outputs["attention_mask"][0].tolist()

        response_ids = input_ids[len(prompt_ids):]
        response_attention_mask = attention_mask[len(prompt_attention_mask):]

        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
        prompt_position_ids = position_ids[:len(prompt_ids)]
        response_position_ids = position_ids[len(prompt_ids):]
        
        # 生成loss mask (仅在response部分计算loss)
        prompt_loss_mask = [0] * len(prompt_ids)
        response_loss_mask = [1] * len(response_ids)
        loss_mask = prompt_loss_mask + response_loss_mask

        out_experience.input_ids = input_ids
        out_experience.prompt_ids = prompt_ids
        out_experience.response_ids = response_ids
        out_experience.attention_mask = attention_mask
        out_experience.prompt_attention_mask = prompt_attention_mask
        out_experience.response_attention_mask = response_attention_mask
        out_experience.position_ids = position_ids
        out_experience.prompt_position_ids = prompt_position_ids
        out_experience.response_position_ids = response_position_ids
        out_experience.loss_mask = loss_mask
        out_experience.prompt_loss_mask = prompt_loss_mask
        out_experience.response_loss_mask = response_loss_mask
        out_experience.reward_scores = rewards_dict
        out_experience.truncate_output_ids()
        # TODO: handle invalid prompt length
        return out_experience


        