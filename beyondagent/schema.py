from uuid import uuid4
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

class Experience(BaseModel):
    """The data model for async rollout."""
    # batch 内sample id，从0->bsz-1
    batch_data_id: int = 0

    # 每一个样本的rollout id，一共n个
    rollout_offset: int = 0

    request_id: str = ""

    messages: List[Dict[str, Any]] = []
    extras: Dict[str, Any] = {}

    # 问题和答案的所有id
    input_ids: List[int] = None
    # 问题的所有id
    prompt_ids: List[int] = None
    # 答案的所有id，可能包括中间的多轮
    response_ids: List[int] = None

    # 和 input_ids 对齐的，决定了是否可见
    attention_mask: List[int] = None

    # 和 prompt_ids 对齐的，都是1
    prompt_attention_mask: List[int] = None

    # 和 response_ids 对齐的，都是1
    response_attention_mask: List[int] = None

    # 和 input_ids 对齐的，rang total len
    position_ids: List[int] = None

    # 和 prompt_ids 对齐的，rang total len
    prompt_position_ids: List[int] = None

    # 和 response_ids 对齐的，rang total len
    response_position_ids: List[int] = None

    # 和 input_ids 对齐的，需要训练的token是1，不需要训练的是0，不训练的包括问题和工具的结果
    loss_mask: List[int] = None

    # 和 prompt_ids 对齐的，需要训练的token是1，不需要训练的是0，不训练的包括问题
    prompt_loss_mask: List[int] = None

    # 和 response_ids 对齐的，需要训练的token是1，不需要训练的是0，不训练的包括答案和工具结果
    response_loss_mask: List[int] = None

    # e.g. {"outcome": 0.0, "description": "Outcome 1 denotes success, and 0 denotes failure."}
    reward_scores: Dict[str, Any] = None
    max_response_len: int = 8192
    max_model_len: int = 32768

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.response_ids[: self.max_response_len]
        self.response_attention_mask = self.response_attention_mask[: self.max_response_len]
        self.response_position_ids = self.response_position_ids[: self.max_response_len]
        self.response_loss_mask = self.response_loss_mask[: self.max_response_len]

    def discard(self) -> None:
        """
        Discard the experience.
        """
        self.input_ids = []
        self.position_ids = []
        self.attention_mask = []
        self.loss_mask = []
        self.prompt_ids = []
        self.response_ids = []
        self.prompt_attention_mask = []
        self.response_attention_mask = []
        self.prompt_position_ids = []
        self.response_position_ids = []
        self.prompt_loss_mask = []
        self.response_loss_mask = []
