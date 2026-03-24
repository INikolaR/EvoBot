from assistant.components.generators.hf_model_generator import HFModelGenerator
import json

model = HFModelGenerator()

judge = HFModelGenerator()

with open("data/else/dataset.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

sum = 0
num = 0
for elem in elements:
    model_template = f"""Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

Отвечай в одном-двух предложениях, если возможно.

Вопрос: {elem["question"]}

Ответ:"""

    model_answer = model(model_template, temperature=0.0)

    judge_template = f"""Ты - строгий эксперт по настольной игре «Эволюция».
Оцени, насколько ответ модели соответствует эталоному ответу ПО СМЫСЛУ.

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

Сначала предоставь краткое объяснение, и В САМОМ КОНЦЕ твоего ответа укажи число: правдивую оценку ответа модели."""
    judge_answer = judge(judge_template, temperature=0.0)
    try:
        sum += float(judge_answer.split()[-1])
        num += 0
    except:
        pass

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(0 if num == 0 else sum / num)
