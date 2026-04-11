from llama_cpp import Llama
from typing import List, Union
import os

class GGUFModelGenerator:
    def __init__(
        self, 
        model_path_or_repo: str = "Qwen/Qwen2.5-32B-Instruct-GGUF",
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        use_4bit: bool = True
    ):
        filename = "*q4_k_m*.gguf" if use_4bit else "*fp16.gguf"

        if os.path.isfile(model_path_or_repo):
            self._llm = Llama(
                model_path=model_path_or_repo,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                chat_format="qwen",
                verbose=False,
                n_threads=max(1, os.cpu_count() - 2)
            )
        else:
            self._llm = Llama.from_pretrained(
                repo_id=model_path_or_repo,
                filename=filename,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                chat_format="qwen",
                verbose=False,
                n_threads=max(1, os.cpu_count() - 2)
            )

    def __call__(self, input_data: Union[str, List[str]], max_new_tokens: int = 1024, temperature: float = 0.1) -> Union[str, List[str]]:
        is_single = isinstance(input_data, str)
        if is_single:
            input_data = [input_data]

        responses = []
        for text in input_data:
            messages = [{"role": "user", "content": text}]
            output = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            responses.append(output["choices"][0]["message"]["content"].strip())

        return responses[0] if is_single else responses
