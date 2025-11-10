import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

with open("knowledge-base-rules.txt", "r") as f:
    kbr = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_text(kbr)

embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

db = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 10,
        "lambda_mult": 0.7
    }
)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
import torch
from langchain.llms.base import LLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

def qwen_generate(prompt):
    print("PROMPT:", prompt)

    inputs_cpu = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs_cpu.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):]

class SimpleQwenLLM(LLM):
    def _call(self, prompt: str, stop = None) -> str:
        return qwen_generate(prompt)

    @property
    def _llm_type(self) -> str:
        return "simple_qwen"

simple_llm = SimpleQwenLLM()

template = """
Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

Используй только предоставленный контекст, чтобы кратко и ясно ответить на вопрос. Не повторяй фрагменты контекста дословно, если не нужно. Отвечай в 1-2 предложениях, если возможно.

Контекст:

{context}

Вопрос: {question}

Ответ:
"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=simple_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

query = "Как работает свойство ЖИРОВОЙ ЗАПАС?"
result = qa_chain.invoke(query)
print(result["result"])
