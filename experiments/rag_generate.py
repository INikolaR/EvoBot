from assistant.pipeline.rag_service import RAGService
from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch

assert torch.cuda.is_available(), "No CUDA provided!"

rag_service = RAGService(
    RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100),
    HFModelEmbedderFactory().create_embedder(hf_model_name="Qwen/Qwen3-Embedding-0.6B"),
    HFModelGenerator("Qwen/Qwen2.5-3B-Instruct"),
    use_rules=True,
    use_faq=True,
    use_comments=True)

with open("data/else/dataset.json", "r", encoding="utf-8") as f:
    elements = json.load(f)

batch_size = 4
json_results = []

for i in range(0, len(elements), batch_size):
    elem_batch = elements[i:min(i+batch_size, len(elements))]

    questions = [elem["question"] for elem in elem_batch]

    result_batch, _ = rag_service.get_response(questions)

    json_result_batch = [{"question" : elem["question"], "model_answer" : result, "reference_answer" : elem["answer"]} for elem, result in zip(elem_batch, result_batch)]

    json_results.extend(json_result_batch)

with open("experiment_results/model_output.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(json_results, ensure_ascii=False, indent=4))
