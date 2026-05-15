from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.chunkers.fixed_chunker import FixedLengthChunker
from assistant.components.chunkers.semantic_chunker import EmbeddingSemanticChunker
import json
import torch
import sys
import time
from langchain_community.retrievers import BM25Retriever
import re

def russian_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\sа-яё]', '', text)
    return text

assert torch.cuda.is_available(), "No CUDA provided!"

chunker_name = sys.argv[1]
output_dir = sys.argv[2]
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
    chunkers.append(EmbeddingSemanticChunker(model_name="Qwen/Qwen3-Embedding-0.6B", threshold_type="percentile", threshold_amount=95))
else:
    print("Unsupported chunker")
    exit(1)

rules_path = "data/documents/rules/knowledge-base-rules.txt"
faq_path = "data/documents/faq/faq.json"
comments_path = "data/documents/comments/comments.json"


for chunker in chunkers:

    with open("data/else/dataset.json", "r", encoding="utf-8") as f:
        elements = json.load(f)

    texts = []
    with open(rules_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    texts += chunker.split_text(raw_text)
    with open(faq_path, "r", encoding="utf-8") as f:
        faq_strings_list = json.load(f)
    texts += faq_strings_list
    with open(comments_path, "r", encoding="utf-8") as f:
        comments = json.load(f)
    useful_comments = []
    for i in range(1, len(comments)):
        if comments[i - 1]["author"] == comments[i]["reply_to"]:
            useful_comments.append(f"Пример пары вопрос-ответ:\nВопрос:\n{comments[i - 1]['text']}\nОтвет:\n{comments[i]['text']}")
    texts += useful_comments

    retriever = BM25Retriever.from_texts(texts, k=3, preprocess_func=russian_preprocess)

    batch_size = 4
    json_results = []

    for i in range(0, len(elements)):
        elem = elements[i]

        question = elem["question"]

        start_time = time.time()
        contexts = retriever.invoke(question)
        end_time = time.time()
        time_diff = (end_time - start_time) / batch_size

        contexts_text = [doc.page_content for doc in contexts]

        json_result = {"question" : elem["question"], "model_contexts" : contexts_text, "time" : time_diff}

        json_results.append(json_result)

    with open(f"{output_dir}/retriever_output_chunker_{chunker.describe()}_embedder_BM25.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(json_results, ensure_ascii=False, indent=4))
