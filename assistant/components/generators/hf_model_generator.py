from assistant.core.generator import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Union

class HFModelGenerator(Generator):
    def __init__(self, hf_model_name: str = "Qwen/Qwen2.5-3B-Instruct", use_4bit: bool = False):
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
        ).eval()

    def __call__(self, input_data: Union[str, List[str]], max_new_tokens: int = 1024, temperature: float = 0.1) -> Union[str, List[str]]:
        is_single = isinstance(input_data, str)
        conversations = []
        if is_single:
            conversations = [[{"role": "user", "content": input_data}]]
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], str):
            conversations = [[{"role": "user", "content": text}] for text in input_data]
        else:
            conversations = input_data

        formatted_texts = self._tokenizer.apply_chat_template(
            conversations, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self._tokenizer(
            formatted_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=8192, 
            padding=True
        )
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
        for i in range(len(conversations)):
            decoded = self._tokenizer.decode(outputs[i][inputs['input_ids'][i].shape[-1]:], skip_special_tokens=True)
            responses.append(decoded.strip())

        return responses[0] if is_single else responses
