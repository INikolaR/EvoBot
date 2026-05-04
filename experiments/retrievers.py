from assistant.pipeline.rag_service import RAGService
from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.chunkers.fixed_chunker import FixedLengthChunker
from assistant.components.chunkers.semantic_chunker import EmbeddingSemanticChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.generators.default_generator import DefaultGenerator
import json
import torch
import sys
import time
import gc

assert torch.cuda.is_available(), "No CUDA provided!"

chunker_name = sys.argv[1]
model_name = sys.argv[2]
output_dir = sys.argv[3]
chunkers = []
if chunker_name == "FixedChunker":
    for chunk_size in [100, 300, 500, 700]:
        for chunk_overlap in [50, 100, 200]:
            if chunk_size <= chunk_overlap:
                continue
            chunkers.append(FixedLengthChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
elif chunker_name == "RecursiveChunker":
    for chunk_size in [100, 300, 500, 700]:
        for chunk_overlap in [50, 100, 200]:
            if chunk_size <= chunk_overlap:
                continue
            chunkers.append(RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
elif chunker_name == "SemanticChunker":
    chunkers.append(EmbeddingSemanticChunker(model_name=model_name, threshold_type="percentile", threshold_amount=95))
else:
    print("Unsupported chunker")
    exit(1)

try:
    model = HFModelEmbedderFactory().create_embedder(hf_model_name=model_name)
except Exception as e:
    print("Unsupported model")
    print(e)
    exit(1)

for chunker in chunkers:
    rag_service = RAGService(
        chunker,
        model,
        DefaultGenerator(),
        use_rules=True,
        use_faq=True,
        use_comments=True)

    with open("data/else/dataset.json", "r", encoding="utf-8") as f:
        elements = json.load(f)

    batch_size = 1
    json_results = []

    for i in range(0, len(elements), batch_size):
        elem_batch = elements[i:min(i+batch_size, len(elements))]

        questions = [elem["question"] for elem in elem_batch]

        start_time = time.time()
        _, context_batch = rag_service.get_response(questions)
        end_time = time.time()
        time_diff = (end_time - start_time) / batch_size

        json_result_batch = [{"question" : elem["question"], "model_contexts" : contexts, "time" : time_diff} for elem, contexts in zip(elem_batch, context_batch)]

        json_results.extend(json_result_batch)

    with open(f"{output_dir}/retriever_output_chunker_{chunker.describe()}_embedder_{model_name.split('/')[-1]}.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(json_results, ensure_ascii=False, indent=4))
    
    del rag_service
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.5)

