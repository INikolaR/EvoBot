from assistant.pipeline.rag_service import RAGService
from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.generators.hf_model_generator import HFModelGenerator
import json
import torch
import gc

assert torch.cuda.is_available(), "No CUDA provided!"

chunkers = []
for chunk_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    for chunk_overlap in [0, 10, 50, 100]:
        chunkers.append(RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100))

print("chunkers created")
embedder_names = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B", "ai-sage/Giga-Embeddings-instruct", "ai-forever/FRIDA", "sergeyzh/BERTA", "intfloat/e5-mistral-7b-instruct"]
for chunker in chunkers[:2]:
    for model_name in embedder_names[:2]:
        print(model_name)
        gc.collect()
        torch.cuda.empty_cache()
        rag_service = RAGService(
            chunker,
            HFModelEmbedderFactory().create_embedder(hf_model_name=model_name),
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

            _, context_batch = rag_service.get_response(questions)

            json_result_batch = [{"question" : elem["question"], "model_context" : context} for elem, context in zip(elem_batch, context_batch)]

            json_results.extend(json_result_batch)

        with open(f"experiment_results/retriever_output_chunker_{chunk_size}_embedder_{model_name.split('/')[-1]}.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(json_results, ensure_ascii=False, indent=4))
