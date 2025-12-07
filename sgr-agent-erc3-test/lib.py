import time
from typing import List, Type, TypeVar

from erc3 import ERC3, TaskInfo
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class MyLLM:
    client: OpenAI
    api: ERC3
    task: TaskInfo
    model: str
    max_tokens: int

    def __init__(self, api: ERC3, model:str, task: TaskInfo, max_tokens=40000) -> None:
        self.api = api
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.client = OpenAI()


    def query(self, messages: List, response_format: Type[T], model: str = None) -> T:

        started = time.time()
        resp = self.client.beta.chat.completions.parse(messages=messages, model=model or self.model, response_format=response_format, max_completion_tokens=self.max_tokens)

        self.api.log_llm(
            task_id=self.task.task_id,
            model=model or self.model,
            duration_sec=time.time() - started,
            completion=resp.choices[0].message.content,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            cached_prompt_tokens=resp.usage.prompt_tokens_details.cached_tokens,
        )

        return resp.choices[0].message.parsed
