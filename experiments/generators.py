from assistant.pipeline.rag_service import RAGService
from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch

assert torch.cuda.is_available(), "No CUDA provided!"

with open("data/else/dataset.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

system_instruction = """Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.
                    
Важно: ответь только на заданный вопрос. Не задавай встречных вопросов, не предлагай продолжить диалог и не генерируй новые темы."""

generator_names = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
for generator_name in generator_names[:2]:

    batch_size = 4
    json_results = []

    for i in range(0, len(elements), batch_size):
        elem_batch = elements[i:min(i+batch_size, len(elements))]

        questions = [elem["question"] for elem in elem_batch]

        system_content = """Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила, ответив на русском языке на его вопрос.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.
                    
Важно: ответь только на заданный вопрос. Не задавай встречных вопросов, не предлагай продолжить диалог и не генерируй новые темы. Для ответа на вопрос используй предоставленный контекст.

Важно: ты отвечаешь ТОЛЬКО на русском языке. Запрещено использовать китайские, английские или иные иностранные слова, символы или фразы. Все ответы должны содержать только кириллицу, цифры и базовые знаки препинания."""

            prompts = [[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"""
Контекст:
{elem["paragraph"]}

Вопрос:
{elem["question"]}
"""}
            ] for elem in elem_batch]

        model_answer_batch = model(prompt_batch, temperature=0.0)

        json_result_batch = [{"question" : elem["question"], "model_answer" : model_answer, "reference_answer" : elem["answer"]} for elem, model_answer in zip(elem_batch, model_answer_batch)]

        json_results.extend(json_result_batch)

    with open(f"experiment_results/generator_output_model_{model_name.split('/')[-1]}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(json_results, ensure_ascii=False, indent=4))
