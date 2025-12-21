from core.generator import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HFModelGenerator(Generator):
    def __init__(self, hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16
        ).eval()

    def __call__(self, input_data: str) -> str:
        prompt = input_data.to_string()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                temperature=0.1
            )

        response = self._tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return response.strip()
