import os
import re
import sys
import numpy as np
import torch
import time

from copy import deepcopy
from typing import Any, Callable, Dict, List
from EnvServiceV1.env.env_client import EnvClient
from recipe.beyond_agent.ba_src.beyondagent_execute import context_generate_from_messages_dummy
from recipe.beyond_agent.schema import Experience
from verl.utils.model import compute_position_id_with_mask
from best_logger import print_dict, print_listofdict
from verl.utils.debug.vscode_breakpoint import vscode_conditional_breakpoint
from loguru import logger
from best_logger import register_logger
non_console_mods = ["appworld_io"]
register_logger(non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path="logs/beyondagent", debug=True)

class TempEnvContextManager():
    def __init__(self, env_service_client, env_type, task_id, instance_id):
        self.env_type = env_type
        self.task_id = task_id
        self.env_service_client = env_service_client
        self.instance_id = instance_id

    def __enter__(self):
        init_response = self.env_service_client.create_instance(self.env_type, self.task_id, self.instance_id)
        self.init_response = init_response
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.env_service_client:
            self.env_service_client.release_instance(self.instance_id)

class EnvAwareEngine:
    def __init__(self, query: Experience, llm_chat_fn: callable, sampling_params: Dict, tokenizer: Any, rank: int = 0):
        self.rank = rank
        self.query = query
        self.llm_chat_fn = llm_chat_fn
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.client = EnvClient()
        while not self.client.check_health(): time.sleep(1); logger.info('Waiting for EnvServiceV1 to be online ...')

    def execute(self):
        traj_messages, rewards_dict = self._run_agent_client(self.query.messages, self.query.extras['task_id'])
        out_experience = self._make_experience(traj_messages, rewards_dict, experience=self.query)
        return out_experience
    
    def parse_llm_call_to_action(self, llm_output: str) -> str:
        llm_output = re.sub(r'<think><think>\n\n</think>\n\n', '', llm_output)
        action = llm_output
        # action = re.findall(r'<tool_call>\n([\s\S]+)\n</tool_call>', llm_output)[0]
        return action
    
    def get_seq_length(self, messages):
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])
    
    def run_step(self, traj_messages, max_seq_length, instance_id, act_step, max_steps):
        llm_output = self.llm_chat_fn(traj_messages)
        traj_messages.extend(llm_output)

        env_output = self.client.step(instance_id, llm_output[0])
        if self.rank == 0:
            print_listofdict(traj_messages, mod='appworld_io', header='LLM Input')
            print_dict(llm_output[0], mod='appworld_io', header='LLM Output')
            print_dict(env_output, mod='appworld_io', header='ENV Output')

        if len(self.tokenizer(env_output["state"]["content"], return_tensors="pt", padding=False)["input_ids"][0]) > 2048:
            _state_content = env_output["state"]["content"]
            truncated_state_content = _state_content[:len(' '.join(_state_content.split()[:2048]))]
            env_output["state"]["content"] = truncated_state_content

        traj_messages.append(env_output["state"])

        return env_output["is_terminated"], env_output["reward"], traj_messages

    def _run_agent_client(self, messages, task_id, **kwargs):
        max_steps = 5
        max_seq_length= 20480
        instance_id = self.query.request_id
        with TempEnvContextManager(self.client, env_type="appworld", task_id=task_id, instance_id=instance_id) as tecm:
            traj_messages = self.query.messages
            for act_step in range(max_steps):
                history_seq_length = self.get_seq_length(traj_messages)
                if history_seq_length > max_seq_length: break
                terminate, reward_score, traj_messages = self.run_step(traj_messages, max_seq_length, instance_id, act_step, max_steps)
                if terminate: break

            if traj_messages[-1]["role"] == "user":
                traj_messages = traj_messages[:-1]

        rewards_dict = {"outcome": reward_score, "description": "Outcome 1 = success, 0 = failure."}
        return traj_messages, rewards_dict

    def _make_experience(self, messages: List, rewards_dict: Dict, experience: Experience):
        out_experience = deepcopy(experience)
        out_experience.messages = messages

        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        outputs = self.tokenizer(full_text, return_tensors="pt", padding=False)
        input_ids = outputs["input_ids"][0].tolist()  # 移除batch维度
        attention_mask = outputs["attention_mask"][0].tolist()
        
        # 只有第一条时prompt，后面都是response
        prompt_text = self.tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        prompt_outputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False)
        prompt_ids = prompt_outputs["input_ids"][0].tolist()
        prompt_attention_mask = prompt_outputs["attention_mask"][0].tolist()

        response_ids = input_ids[len(prompt_ids):]
        response_attention_mask = attention_mask[len(prompt_attention_mask):]

        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
        prompt_position_ids = position_ids[:len(prompt_ids)]
        response_position_ids = position_ids[len(prompt_ids):]
        
        # 生成loss mask (仅在response部分计算loss，但需要在response部分mask env的输出)
        prompt_loss_mask = [0] * len(prompt_ids)
        response_loss_mask = [1] * len(response_ids)

        response_token_ids_idxs = []
        human_token_ids_idxs = []

        response_ids_np = np.array(response_ids)

        for assistant_idx in np.where(response_ids_np == self.response_template_ids[0])[0]:
            if (self.response_template_ids == response_ids_np[assistant_idx: assistant_idx + len(self.response_template_ids)].tolist()):
                response_token_ids_idxs.append(assistant_idx + len(self.response_template_ids))

        for human_idx in np.where(response_ids_np == self.instruction_template_ids[0])[0]:
            if self.instruction_template_ids == response_ids_np[human_idx: human_idx + len(self.instruction_template_ids)].tolist():
                human_token_ids_idxs.append(human_idx)

        if (
            len(human_token_ids_idxs) > 0
            and len(response_token_ids_idxs) > 0
            and human_token_ids_idxs[0] > response_token_ids_idxs[0]
        ):
            human_token_ids_idxs = [0] + human_token_ids_idxs

        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            response_loss_mask[start:end] = [0] * (end-start)

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            response_loss_mask[human_token_ids_idxs[-1]:] = [0] * (len(response_loss_mask)-human_token_ids_idxs[-1])

        # print(response_loss_mask)

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


                