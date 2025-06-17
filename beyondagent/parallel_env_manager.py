import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from recipe.beyond_agent.env_aware_engine import EnvAwareEngine
from copy import deepcopy
from typing import Any, Dict, List
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.import_utils import load_extern_type
from verl.utils.torch_functional import (get_response_mask,
                                         pad_sequence_to_length)
from verl.workers.rollout.async_server import AsyncLLMServerManager

from .callback import simple_callback
from .schema import Experience
from loguru import logger


class ParallelEnvManager:
    def __init__(self, config: DictConfig, async_rollout_manager: AsyncLLMServerManager, max_parallel: int = 128):
        self.config = config
        self.async_rollout_manager = async_rollout_manager
        self.max_parallel = max_parallel

        self.n = config.actor_rollout_ref.rollout.n
        self.model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])

        self.tokenizer = self.async_rollout_manager.chat_scheduler.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        
        
        rollout_config = config.actor_rollout_ref.rollout
        self.rollout_config = rollout_config


    def get_llm_chat_fn(self, sampling_params: Dict[str, Any]={}) -> callable:

        def llm_chat(messages: List[Dict[str, str]], custom_sampling_params: Dict[str, Any]={}) -> List[Dict[str, Any]]:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = deepcopy(sampling_params)
            updated_sampling_params.update(custom_sampling_params)

            output_messages = []
            self.async_rollout_manager.submit_chat_completions(
                    callback=simple_callback,
                    callback_additional_info={"output_messages": output_messages},
                    model=self.model_name,
                    messages=messages,
                    **updated_sampling_params
                )
            return output_messages
        
        return llm_chat

    def rollout_single_env_engine(self, prompt: Experience, do_sample: bool, is_validate: bool, idx: int) -> Experience:
        """
        Process a single prompt in a thread-safe way.
        """
        #### TODO: update sampling params according to the mode
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.rollout_config.response_length,
            temperature=self.rollout_config.temperature,
            top_p=self.rollout_config.top_p,
        )
        if not do_sample or is_validate:
            sampling_params["temperature"] = self.rollout_config.val_kwargs.temperature
            sampling_params["top_k"] = self.rollout_config.val_kwargs.top_k
            sampling_params["top_p"] = self.rollout_config.val_kwargs.top_p

        dataflow = EnvAwareEngine(
            query=prompt,
            llm_chat_fn=self.get_llm_chat_fn(sampling_params),
            sampling_params=sampling_params,
            tokenizer=self.tokenizer,
            rank=idx,
        )
        output_exp: Experience = dataflow.execute()
        return output_exp

    def rollout(self, gen_batch: DataProto, **kwargs) -> DataProto:
        """
        Process prompts in parallel using a thread pool.

        Args:
            gen_batch (DataProto): Input batch containing prompts.

        Returns:
            DataProto: Combined output of all processed prompts.
        """
        do_sample = gen_batch.meta_info.get("do_sample", True)
        is_validate = gen_batch.meta_info.get("validate", False)

        prompt_dicts: List[Experience] = self._extract_prompts_from_dataproto(
                                                gen_batch, 
                                                n=1 if is_validate else self.rollout_config.n
                                            )
        num_prompts = len(prompt_dicts)
        output_experience_list = []

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = []
            for idx, prompt in enumerate(prompt_dicts):
                future = executor.submit(self.rollout_single_env_engine, prompt, do_sample, is_validate, idx)
                futures.append(future)

            for future in futures:
                # do not fail silently
                result = future.result()
                output_experience_list.append(result)

        # Sort results by index to preserve input order
        sorted_output_experience_list = sorted(output_experience_list, key=lambda x: (x.batch_data_id, x.rollout_offset))

        # print("Total length of experience list", len(sorted_output_experience_list))
        # print("Show an experience from dataflow: \n", sorted_output_experience_list[0])
        # print("Decoded response: ", self.tokenizer.decode(sorted_output_experience_list[0].response_ids))
        # TODO: from yunpeng
        # incorporate the haoyu dataflow with env
        # Combine and pad
        data_proto = self._pad_and_cat_to_dataproto(sorted_output_experience_list)
        return data_proto


    def _extract_prompts_from_dataproto(self, prompts: DataProto, n: int) -> List[Experience]:
        assert "raw_prompt" in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        exp_list = []

        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for rollout_offset in range(n):
                exp = Experience(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    messages=[msg for msg in raw_prompt],
                    extras=prompts.non_tensor_batch["extras"][data_idx] if "extras" in prompts.non_tensor_batch else {},
                    max_response_len=self.rollout_config.response_length,
                    max_model_len=self.rollout_config.prompt_length + self.rollout_config.response_length
                )
                exp_list.append(exp)

        return exp_list
    
    def _pad_and_cat_to_dataproto(self, experiences: list[Experience]):
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        reward_scores = []
        for exp in experiences:
            assert len(exp.input_ids) == len(exp.attention_mask) == len(exp.position_ids) == len(exp.loss_mask), f"""Experience {exp.request_id} has different length of 
                {len(exp.input_ids)=}, {len(exp.attention_mask)=}, {len(exp.position_ids)=}, {len(exp.loss_mask)=}"""
            error_message_lines = [
                f"""Request {exp.request_id} has input_ids length {len(exp.input_ids)}
                    greater than max_model_len {self.rollout_config.max_model_len}""",
                f"Decoded input_ids: {self.tokenizer.decode(exp.input_ids)}",
                f"Decoded prompt_ids: {self.tokenizer.decode(exp.prompt_ids)}",
                f"Decoded response_ids: {self.tokenizer.decode(exp.response_ids)}",
                f"Messages: {exp.messages}",
                f"Max model length: {exp.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            # assert len(exp.input_ids) <= self.rollout_config.max_model_len, error_message
            if len(exp.prompt_ids) > self.rollout_config.prompt_length:
                # empty this experience
                logger.warning(
                    f"""{exp.request_id=} has prompt_ids length {len(exp.prompt_ids)} 
                    greater than max_prompt_length {self.rollout_config.prompt_length},\n{exp=}"""
                )
                exp.discard()

            prompt_ids.append(torch.tensor(exp.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(exp.response_ids, dtype=torch.int))
            if len(exp.response_ids) > self.rollout_config.response_length:
                logger.warning(
                    f"""{exp.request_id=} has response_ids length {len(exp.response_ids)} 
                    greater than max_response_len {self.rollout_config.response_length},\n{exp=}"""
                )
            prompt_attention_mask.append(torch.tensor(exp.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(exp.response_attention_mask, dtype=torch.int))
            prompt_position_ids.append(torch.tensor(exp.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(exp.response_position_ids, dtype=torch.int))
            prompt_loss_mask.append(torch.tensor(exp.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(exp.response_loss_mask, dtype=torch.int))
            messages.append({"messages": exp.messages})
            reward_scores.append(exp.reward_scores)

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        if prompt_ids.shape[1] < self.rollout_config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.rollout_config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.rollout_config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.rollout_config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_attention_mask.shape[1] < self.rollout_config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.rollout_config.prompt_length, 0, left_pad=True)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.rollout_config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.rollout_config.response_length, 0)
        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        if prompt_position_ids.shape[1] < self.rollout_config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.rollout_config.prompt_length, 0, left_pad=True)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(len(experiences), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.rollout_config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.rollout_config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.rollout_config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.rollout_config.response_length, 0)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(experiences),
        )

        return DataProto(batch=batch, non_tensor_batch={"messages": np.array(messages), "reward_scores": np.array(reward_scores)})