import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

with open("knowledge-base-rules.txt", "r", encoding="utf-8") as f:
    kbr = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_text(kbr)
docs = [{"content": t} for t in texts]

embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

db = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
retriever = db.as_retriever(
    search_kwargs={
        "k": 2,
        "fetch_k": 10,
        "lambda_mult": 0.7
    },
    search_type="mmr"
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

def qwen_generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.1
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response.strip()

class CustomLLM:
    def invoke(self, input_data):
        prompt = input_data.get("text") or input_data
        return qwen_generate(prompt)

    def __call__(self, input_data):
        return self.invoke(input_data)

llm = CustomLLM()

template = """
Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

Используй только предоставленный контекст, чтобы кратко и ясно ответить на вопрос. Не повторяй фрагменты контекста дословно, если не нужно. Отвечай в 1-2 предложениях, если возможно.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc['content'] for doc in docs])

rag_chain_from_docs = (
    {
        "context": lambda x: format_docs(x),
        "question": lambda x: x[0].metadata.get('question', '') if x and hasattr(x[0], 'metadata') else ''
    }
    | QA_PROMPT
    | llm
    | StrOutputParser()
)

def get_rag_response(question: str):
    retrieved_docs = retriever.get_relevant_documents(question)
    for doc in retrieved_docs:
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        doc.metadata['question'] = question
    result = rag_chain_from_docs.invoke(retrieved_docs)
    return result

query = "Как работает свойство ЖИРОВОЙ ЗАПАС?"
response = get_rag_response(query)
print(response)
