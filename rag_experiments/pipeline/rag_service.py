from rag_experiments.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from rag_experiments.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from rag_experiments.components.retrievers.chroma_retriever_factory import ChromaRetrieverFactory
from rag_experiments.components.generators.hf_model_generator import HFModelGenerator
from rag_experiments.core.chunker import Chunker
from rag_experiments.core.generator import Generator
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

class RAGService:
    def __init__(self, knowledge_base_path: str, chunker: Chunker = RecursiveCharacterChunker(), embedder: Embeddings = HFModelEmbedderFactory().create_embedder(), generator: Generator = HFModelGenerator()):
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = self._init_retriever(knowledge_base_path)
        self.generator = generator
        self.prompt_template = self._init_prompt()
        self.rag_chain_from_docs = self._init_rag_chain_from_docs()

    def _init_retriever(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        texts = self.chunker.split_text(raw_text)
        return ChromaRetrieverFactory().create_retriever(
            texts, self.embedder,
            search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.7},
            search_type="mmr"
        )

    def _init_prompt(self):
        from langchain_core.prompts import PromptTemplate
        template = """Ты - консультант по серии настольных игр "Эволюция"..."""
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def _init_rag_chain_from_docs(self):
        template = """Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

        Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

        Используй только предоставленный контекст, чтобы кратко и ясно ответить на вопрос. Не повторяй фрагменты контекста дословно, если не нужно. Отвечай в 1-2 предложениях, если возможно.

        Контекст:
        {context}

        Вопрос: {question}

        Ответ:"""

        qa_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return (
            {
                "context": lambda x: RAGService._format_docs(x),
                "question": lambda x: x[0].metadata.get('question', '') if x and hasattr(x[0], 'metadata') else ''
            }
            | qa_prompt
            | self.generator
            | StrOutputParser()
        )

    def get_response(self, question: str) -> tuple[str, str]:
        retrieved_docs = self.retriever.invoke(question)
        for doc in retrieved_docs:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['question'] = question
        result = self.rag_chain_from_docs.invoke(retrieved_docs)
        return result.split("\n")[0], RAGService._format_docs(retrieved_docs)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n".join([doc.page_content if hasattr(doc, 'page_content') else doc['content'] for doc in docs])