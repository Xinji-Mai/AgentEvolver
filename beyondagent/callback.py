import asyncio
import json
from typing import Any, Dict

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
async def simple_callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
    assert exception is None, f"exception: {exception}"
    messages = info["output_messages"]
    message = completions.choices[0].message
    messages.append({"role": message.role, "content": message.content})
    # print(f"[round={round}] role: {message.role}, content: {message.content}")
