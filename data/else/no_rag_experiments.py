from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch

assert torch.cuda.is_available(), "No CUDA provided!"

model = HFModelGenerator("Qwen/Qwen2.5-14B-Instruct")
judge = HFModelGenerator("Qwen/Qwen2.5-14B-Instruct")

with open("data/else/dataset.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

total_score = 0.0
count = 0

print("[")

for elem in elements:
    model_template = f"""Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

Отвечай в одном-двух предложениях, если возможно.

Вопрос: {elem["question"]}

Ответ:"""

    model_answer = model(model_template, temperature=0.0)

    judge_template = f"""Ты - строгий эксперт по настольной игре «Эволюция».
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
{elem["answer"]}

ОТВЕТ МОДЕЛИ:
{model_answer}

Предоставь ответ СТРОГО В ВИДЕ JSON (без лишнего текста):
{{
  "score": число 0.0-1.0,
  "reason": "кратко на русском"
}}"""
    
    judge_answer = judge(judge_template, temperature=0.0, max_new_tokens=128)

    o = {"question" : elem["question"], "model_answer" : model_answer, "reference_answer" : elem["answer"], "judge_feedback" : judge_answer, "grade" : None}
    
    try:
        data = json.loads(judge_answer[judge_answer.find("{"):judge_answer.find("}")+1])
        o["grade"] = data
        score = float(data["score"])
        
        total_score += score
        count += 1
    except Exception as e:
        print(f"Ошибка парсинга ответа судьи: {e}")
        pass
    print(json.dumps(o, ensure_ascii=False, indent=4) + ",")
print("]")


result = 0.0 if count == 0 else total_score / count
print(str(result))
