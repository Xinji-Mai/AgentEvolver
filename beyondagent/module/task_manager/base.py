import abc
from typing import Any, Protocol

from beyondagent.schema.task import Task, TaskObjective


class LlmClient(Protocol):
    def chat(
        self, messages: list[dict[str, str]], sampling_params: dict[str, Any]
    ) -> str: ...

class LlmRawClient(Protocol):
    def chat(
        self, messages: list[dict[str, str]], sampling_params: dict[str, Any]
    ) -> dict: ...

class TaskObjectiveRetrieval(abc.ABC):
    """支持任务相关任务 objective 检索，用于避免重复探索"""

    @abc.abstractmethod
    def retrieve_objectives(self, task: Task) -> list[TaskObjective]: ...

    @abc.abstractmethod
    def add_objective(self, objective: TaskObjective): ...
    
    @abc.abstractmethod
    def reset(self):...



class NaiveTaskObjectiveRetrieval(TaskObjectiveRetrieval):

    def __init__(self):
        # 目前单次训练中只会有同一个 env_type 的 task，所以可以直接使用 task_id as key
        self._mp: dict[str, list[TaskObjective]] = {}

    def retrieve_objectives(self, task: Task) -> list[TaskObjective]:
        if task.task_id not in self._mp:
            return []
        return self._mp[task.task_id]

    def add_objective(self, objective: TaskObjective):
        """
        Adds a new task objective to the internal mapping. If the task ID does not exist in the mapping, it initializes an empty list for that task ID.

        Args:
            objective (TaskObjective): The task objective to be added.
        """
        if objective.task.task_id not in self._mp:
            self._mp[objective.task.task_id] = []  # ⭐ Initialize an empty list for the task ID if it doesn't exist

        self._mp[objective.task.task_id].append(objective)  # ⭐ Add the objective to the list for the task ID

    def reset(self):
        """
        Clears the internal mapping, removing all stored task objectives.
        """
        self._mp = {}  # ⭐ Clear the internal mapping