from assistant.core.generator import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Union

class HFModelGenerator(Generator):
    def __init__(self, hf_model_name: str = "Qwen/Qwen2.5-3B-Instruct", device_id: int = 0):
        self._device = f"cuda:{device_id}"
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            device_map={"": self._device},
            dtype=torch.float16#,
            # load_in_8bit=True
        ).eval()

    def __call__(self, input_data: Union[str, List[str]], max_new_tokens: int = 128, temperature: int = 0.1) -> Union[str, List[str]]:
        is_single = isinstance(input_data, str)
        prompts = [input_data] if is_single else input_data

        inputs = self._tokenizer(prompts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                temperature=temperature if temperature > 0 else 1.0
            )

        responses = []
        for i in range(len(prompts)):
            decoded = self._tokenizer.decode(outputs[i][inputs['input_ids'][i].shape[-1]:], skip_special_tokens=True)
            responses.append(decoded.strip())

        return responses[0] if is_single else responses
