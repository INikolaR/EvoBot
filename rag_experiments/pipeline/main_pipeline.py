from rag_experiments.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from rag_experiments.components.embedders.default_embedder_factory import DefaultEmbedderFactory
from rag_experiments.components.retrievers.chroma_retriever_factory import ChromaRetrieverFactory
from rag_experiments.components.generators.default_generator import DefaultGenerator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

with open("research/knowledge-base-rules.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

chunker = RecursiveCharacterChunker()
texts = chunker.split_text(raw_text)

embedder = DefaultEmbedderFactory().create_embedder()

retriever = ChromaRetrieverFactory().create_retriever(
    texts,
    embedder,
    search_kwargs={
        "k": 2,
        "fetch_k": 10,
        "lambda_mult": 0.7
    },
    search_type="mmr")

generator = DefaultGenerator()

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
    | generator
    | StrOutputParser()
)

def get_rag_response(question: str):
    retrieved_docs = retriever.invoke(question)
    for doc in retrieved_docs:
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        doc.metadata['question'] = question
    result = rag_chain_from_docs.invoke(retrieved_docs)
    return result

if __name__ == '__main__':
    query = "Как работает свойство МИМИКРИЯ?"
    response = get_rag_response(query)
    print(response)
