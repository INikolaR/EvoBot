from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch

assert torch.cuda.is_available(), "No CUDA provided!"

judge = HFModelGenerator("Qwen/Qwen2.5-14B-Instruct")

with open("experiment_results/model_output.txt", "r", encoding="utf-8") as f:
    elements = json.load(f) # {"question" : "", "model_answer" : "", "reference_answer" : ""}

total_score = 0
count = 0
batch_size = 4
json_results = []

for i in range(0, len(elements), batch_size):
    elem_batch = elements[i:min(i+batch_size, len(elements))]
    prompt_batch = [f"""Ты - строгий эксперт по настольной игре «Эволюция».
Оцени, насколько ответ модели соответствует эталонному ответу ПО СМЫСЛУ.

КРИТЕРИИ:
- 1.0: полностью эквивалентен по смыслу
- 0.8-0.99: верен, но упущена мелочь
- 0.5-0.79: частично верен, есть важные упущения
- 0.0-0.49: неверен или содержит выдумки

ВАЖНО: Оценивай смысл, а не дословное совпадение.

ВОПРОС:
{elem["question"]}

ЭТАЛОННЫЙ ОТВЕТ:
{elem["reference_answer"]}

ОТВЕТ МОДЕЛИ:
{elem["model_answer"]}

Предоставь ответ СТРОГО В ВИДЕ JSON (без лишнего текста):
{{
  "score": число 0.0-1.0,
  "reason": "кратко на русском"
}}""" for elem in elem_batch]

    judge_answer_batch = judge(prompt_batch, temperature=0.0)

    for elem, judge_answer in zip(elem_batch, judge_answer_batch):
        o = {"question" : elem["question"], "model_answer" : elem["model_answer"], "reference_answer" : elem["reference_answer"], "judge_feedback" : judge_answer, "grade" : None}
        try:
            data = json.loads(judge_answer[judge_answer.find("{"):judge_answer.find("}")+1])
            o["grade"] = data
            score = float(data["score"])
            
            total_score += score
            count += 1
        except Exception as e:
            print(f"Ошибка парсинга ответа судьи: {e}")
            pass
        json_results.append(o)
with open("experiment_results/judge_output.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(json_results, ensure_ascii=False, indent=4))

result = 0.0 if count == 0 else total_score / count
print(str(result))