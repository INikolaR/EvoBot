from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch

assert torch.cuda.is_available(), "No CUDA provided!"

model = HFModelGenerator("Qwen/Qwen2.5-3B-Instruct")

with open("data/else/dataset.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

batch_size = 4
json_results = []

for i in range(0, len(elements), batch_size):
    elem_batch = elements[i:min(i+batch_size, len(elements))]
    prompt_batch = [f"""Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

Отвечай в одном-двух предложениях, если возможно.

Вопрос: {elem["question"]}

Ответ:""" for elem in elem_batch]

    model_answer_batch = model(prompt_batch, temperature=0.0)

    json_result_batch = [{"question" : elem["question"], "model_answer" : model_answer, "reference_answer" : elem["answer"]} for elem, model_answer in zip(elem_batch, model_answer_batch)]

    json_results.extend(json_result_batch)

with open("experiment_results/model_output.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(json_results, ensure_ascii=False, indent=4))
