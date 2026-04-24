from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch
import glob
import os

assert torch.cuda.is_available(), "No CUDA provided!"

judge = HFModelGenerator("Qwen/Qwen2.5-14B-Instruct")

directory = "experiment_results"

txt_files = glob.glob(os.path.join(directory, "*.txt"))

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
        elements = json.load(f) # {"question" : "", "model_context" : ""}

    total_score = 0
    count = 0
    error_count = 0
    batch_size = 4
    json_results = []

    system_instruction = """Ты - строгий оценщик релевантности текстовых фрагментов для поисковой системы.
    Задача: определи, содержит ли ФРАГМЕНТ ПРАВИЛ информацию, необходимую для ответа на ВОПРОС.

    КРИТЕРИИ:
    - TRUE: Фрагмент содержит факты, правила или пояснения, прямо отвечающие на вопрос или являющиеся обязательной частью ответа, даже если информация сформулирована иначе, но передаёт тот же смысл.
    - FALSE: Фрагмент не содержит нужной информации, даёт общие сведения или относится к другой теме.

    ВАЖНО:
    1. Оценивай ТОЛЬКО по ТЕКСТУ. Внешние знания игнорируй.
    2. Если фрагмент содержит лишь часть нужной информации, но без неё ответ невозможен - ставь TRUE.
    3. Ответ строго в формате JSON: {"reason": "кратко на русском", "relevant": "true"/"false"}
    """

    for i in range(0, len(elements), batch_size):
        elem_batch = elements[i:min(i+batch_size, len(elements))]
        
        prompt_batch = [
            [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"""ВОПРОС:
    {elem["question"]}

    ФРАГМЕНТ ПРАВИЛ:
    {elem["model_context"]}

    ОТВЕТ ЖЮРИ (НАЧНИ СТРОГО С {"{"}):
    """}
            ] 
            for elem in elem_batch
        ]

        judge_answer_batch = judge(prompt_batch, temperature=0.0, max_new_tokens=1024)

        for elem, judge_answer in zip(elem_batch, judge_answer_batch):
            o = {"question" : elem["question"], "model_context" : elem["model_context"], "judge_feedback" : judge_answer, "grade" : None}
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
                score = 0.0 if o["grade"]["score"] == "false" else 1.0
                
                total_score += score
                count += 1
            except Exception as e:
                print(f"Ошибка парсинга ответа судьи: {e}")
                error_count += 1
                pass
            json_results.append(o)
    with open("experiment_results/judge_outputs/" + file_path.split('/')[-1], "w", encoding="utf-8") as f:
        f.write(json.dumps(json_results, ensure_ascii=False, indent=4))

    with open("experiment_results/reports/" + file_path.split('/')[-1] + "_report.txt", "w", encoding="utf-8") as f:
        f.write("items count: " + str(count + error_count) + "\n")
        f.write("error count: " + str(error_count) + "\n")
        f.write("error ratio: " +  str(0.0 if count + error_count == 0 else error_count / (count + error_count)) + "\n")
        f.write("score: " + str(0.0 if count == 0 else total_score / count) + "\n")
