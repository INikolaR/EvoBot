from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch
import glob
import os
import sys

assert torch.cuda.is_available(), "No CUDA provided!"

input_folder = sys.argv[1]
output_folder = sys.argv[2]
report_folder = sys.argv[3]

judge = HFModelGenerator("Qwen/Qwen2.5-14B-Instruct")

txt_files = glob.glob(os.path.join(input_folder, "*.txt"))

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
        elements = json.load(f) # {"question" : "", "model_contexts" : ["", "", ""]}

    print("working with file", file_path, flush=True)

    sum_reciprocal_rank = 0
    sum_mean_precision = 0
    count = 0
    error_count = 0
    batch_size = 4
    n_chunks_retrieved = 3
    json_results = []

    system_instruction = """Ты - строгий оценщик релевантности текстовых фрагментов для поисковой системы.
    Задача: определи, содержит ли ФРАГМЕНТ ПРАВИЛ информацию, необходимую для ответа на ВОПРОС.

    КРИТЕРИИ:
    - 1.0: Фрагмент содержит факты, правила или пояснения, прямо отвечающие на вопрос или являющиеся обязательной частью ответа, даже если информация сформулирована иначе, но передаёт тот же смысл.
    - 0.0: Фрагмент не содержит нужной информации, даёт общие сведения или относится к другой теме.

    ВАЖНО:
    1. Оценивай ТОЛЬКО по ТЕКСТУ. Внешние знания игнорируй.
    2. Если фрагмент содержит лишь часть нужной информации, но без неё ответ невозможен - ставь 1.0.
    3. Ответ строго в формате JSON: {"reason": "кратко на русском", "relevant": 1.0/0.0}
    """

    for i in range(0, len(elements), batch_size):
        elem_batch = elements[i:min(i+batch_size, len(elements))]

        print(f"processing elements [{i}, {min(i+batch_size, len(elements)) - 1}]", flush=True)
        
        #stores in format [[context1 for q1, context1 for q2, context1 for q3, ...], [context2 for q1, context2 for q2, context2 for q3, ...], ...]
        judge_answer_batches = []

        for j in range(n_chunks_retrieved):
            prompt_batch = [
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"""ВОПРОС:
{elem["question"]}

ФРАГМЕНТ ПРАВИЛ:
{elem["model_contexts"][j]}

ОТВЕТ ЖЮРИ (НАЧНИ СТРОГО С {"{"}):
"""}
                ] 
                for elem in elem_batch
            ]

            judge_answer_batch = judge(prompt_batch, temperature=0.0, max_new_tokens=1024)
            judge_answer_batches.append(judge_answer_batch)

        #stores in format [[context1 for q1, context2 for q1, context3 for q1, ...], [context1 for q2, context2 for q2, context3 for q2, ...], ...]
        judge_answer_batches_per_questions = [list(ans_tuple) for ans_tuple in zip(*judge_answer_batches)]        
        
        for elem, judge_answers in zip(elem_batch, judge_answer_batches_per_questions):
            o = {"question" : elem["question"], "model_contexts" : elem["model_contexts"], "judge_feedbacks" : judge_answers, "grade" : [None for _ in range(n_chunks_retrieved)]}
            try:
                if len(o["model_contexts"]) > 0:
                    for j in range(n_chunks_retrieved):
                        judge_answer = judge_answers[j]
                        end = judge_answer.find("}")
                        start = judge_answer.rfind('{', 0, end)
                        expected_json = judge_answer[start:end+1]
                        count_quotes = expected_json.count("\"")
                        expected_json_without_quotes_in_reason = expected_json.replace("\"", "\'", count_quotes - 3).replace("\'", "\"", 3)
                        o["grade"][j] = json.loads(expected_json)
                else:
                    o["grade"] = [{"reason" : "ответ отсутствует", "relevant" : 0.0} for _ in range(n_chunks_retrieved)]

                rank = 0
                sum_score = 0
                for j in range(n_chunks_retrieved, 0, -1):
                    score = float(o["grade"][j - 1]["relevant"])
                    if score > 0.5:
                        rank = j
                    sum_score += score
                sum_reciprocal_rank += 1 / rank if rank > 0 else 0

                sum_mean_precision += sum_score / n_chunks_retrieved
                
                count += 1
            except Exception as e:
                print(f"Ошибка парсинга ответа судьи: {e}")
                error_count += 1
                pass
            json_results.append(o)
    with open(f"{output_folder}/" + file_path.split('/')[-1], "w", encoding="utf-8") as f:
        f.write(json.dumps(json_results, ensure_ascii=False, indent=4))

    with open(f"{report_folder}/" + file_path.split('/')[-1] + "_report.txt", "w", encoding="utf-8") as f:
        f.write("items count: " + str(count + error_count) + "\n")
        f.write("error count: " + str(error_count) + "\n")
        f.write("error ratio: " +  str(0.0 if count + error_count == 0 else error_count / (count + error_count)) + "\n")
        f.write("mrr: " + str(sum_reciprocal_rank / count) + "\n")
        f.write("p@3: " + str(sum_mean_precision / count) + "\n")
