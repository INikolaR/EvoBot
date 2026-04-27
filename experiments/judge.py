from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch
import glob
import os

assert torch.cuda.is_available(), "No CUDA provided!"

judge = HFModelGenerator("Qwen/Qwen2.5-32B-Instruct")

directory = "experiment_results"

txt_files = glob.glob(os.path.join(directory, "*.txt"))

for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as f:
        elements = json.load(f) # {"question" : "", "model_answer" : "", "reference_answer" : "", "model_context" : "", "reference_context" : ""}

    total_score = 0
    count = 0
    error_count = 0
    batch_size = 4
    json_results = []

    system_instruction = """Ты - строгий эксперт по настольной игре «Эволюция».
Твоя задача: оценить, насколько ответ модели соответствует эталонному ответу и фрагменту правил игры ПО СМЫСЛУ.

ТЫ ДОЛЖЕН ПРЕДОСТАВИТЬ ОТВЕТ СТРОГО В ФОРМАТЕ JSON:
{
"reason": "кратко на русском",
"score": число 0.0-1.0
}

КРИТЕРИИ:
- 1.0: содержит все детали из эталонного ответа и не противоречит указанному фрагменту правил
- 0.8-0.99: упускает мелочь по сравнению с эталонным ответом или слегка противоречит указанному фрагменту правил
- 0.5-0.79: есть значимые упущения по сравнению с эталонным ответом или есть серьёзные противоречия фрагменту правил
- 0.0-0.49: неверен или содержит много информации, не относящейся к правилам

ВАЖНО:
1. Ты должен оценивать смысл, а не дословное совпадение.
2. Ты должен строго следовать критериям оценки.
3. Ты должен точно соблюдать формат ответа в виде JSON.
"""

    for i in range(0, len(elements), batch_size):
        elem_batch = elements[i:min(i+batch_size, len(elements))]
        
        prompt_batch = [
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"""ВОПРОС:
{elem["question"]}

ЭТАЛОННЫЙ ОТВЕТ:
{elem["reference_answer"]}

ФРАГМЕНТ ПРАВИЛ:
{elem["reference_context"]}

ОТВЕТ МОДЕЛИ:
{elem["model_answer"]}

ОТВЕТ ЖЮРИ (НАЧНИ СТРОГО С {"{"}):
"""}
            ] 
            for elem in elem_batch
        ]

        judge_answer_batch = judge(prompt_batch, temperature=0.0, max_new_tokens=1024)

        for elem, judge_answer in zip(elem_batch, judge_answer_batch):
            o = {"question" : elem["question"], "model_answer" : elem["model_answer"], "reference_answer" : elem["reference_answer"], "reference_context" : elem["reference_context"], "judge_feedback" : judge_answer, "grade" : None}
            try:
                if len(o["model_answer"]) > 0:
                    end = judge_answer.find("}")
                    start = judge_answer.rfind('{', 0, end)
                    expected_json = judge_answer[start:end+1]
                    count_quotes = expected_json.count("\"")
                    expected_json_without_quotes_in_reason = expected_json.replace("\"", "\'", count_quotes - 3).replace("\'", "\"", 3)
                    o["grade"] = json.loads(expected_json_without_quotes_in_reason)
                else:
                    o["grade"] = {"score" : 0.0, "reason" : "ответ отсутствует"}
                score = float(o["grade"]["score"])
                
                total_score += score
                count += 1
            except Exception as e:
                print(f"Ошибка парсинга ответа судьи: {e}")
                error_count += 1
                pass
            json_results.append(o)
    with open(f"experiment_results/judge_outputs/" + txt_file.split('/')[-1], "w", encoding="utf-8") as f:
        f.write(json.dumps(json_results, ensure_ascii=False, indent=4))

    with open("experiment_results/reports/" + txt_file.split('/')[-1], "w", encoding="utf-8") as f:
        f.write("question count: " + str(count + error_count) + "\n")
        f.write("error count: " + str(error_count) + "\n")
        f.write("error ratio: " +  str(0.0 if count + error_count == 0 else error_count / (count + error_count)) + "\n")
        f.write("score: " + str(0.0 if count == 0 else total_score / count) + "\n")
