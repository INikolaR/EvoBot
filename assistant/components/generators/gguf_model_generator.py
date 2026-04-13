from llama_cpp import Llama
from typing import List, Union
import os
import glob
from huggingface_hub import snapshot_download

class GGUFModelGenerator:
    def __init__(
        self, 
        model_path_or_repo: str = "Qwen/Qwen2.5-32B-Instruct-GGUF",
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        use_4bit: bool = True
    ):
        if os.path.isfile(model_path_or_repo):
            model_path = model_path_or_repo
        else:
            pattern = "*q4_k_m*.gguf" if use_4bit else "*fp16*.gguf"

            download_dir = snapshot_download(
                repo_id=model_path_or_repo,
                allow_patterns=pattern,
                ignore_patterns=["*.md", "*.txt", "*.json", "*.png"]
            )

            downloaded_files = sorted(glob.glob(os.path.join(download_dir, pattern)))
            
            if not downloaded_files:
                raise FileNotFoundError(f"Файлы по паттерну {pattern} не найдены в {model_path_or_repo}")

            model_path = downloaded_files[0]

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=4096,
            flash_attn=True,
            offload_kqv=True,
            n_threads=1,
            chat_format="qwen",
            verbose=False,
        )

    def __call__(self, input_data: List[dict], max_new_tokens: int = 1024, temperature: float = 0.1) -> Union[str, List[str]]:
        responses = []
        for dialog in input_data:
            output = self._llm.create_chat_completion(
                messages=dialog,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            responses.append(output["choices"][0]["message"]["content"].strip())

        return responses
