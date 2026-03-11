from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.retrievers.chroma_retriever_factory import ChromaRetrieverFactory
from assistant.components.generators.hf_model_generator import HFModelGenerator
from assistant.core.chunker import Chunker
from assistant.core.generator import Generator
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json

class RAGService:
    def __init__(self, chunker: Chunker = None, embedder: Embeddings = None, generator: Generator = None):
        self.chunker = chunker if chunker is not None else RecursiveCharacterChunker()
        self.embedder = embedder if embedder is not None else HFModelEmbedderFactory().create_embedder()
        self.retriever = self._init_retriever("data/documents//rules/knowledge-base-rules.txt",
                                              "data/documents/faq/faq.json",
                                              "data/documents/comments/comments.json")
        self.generator = generator if generator is not None else HFModelGenerator()
        self.prompt_template = self._init_prompt()
        self.rag_chain_from_docs = self._init_rag_chain_from_docs()

    def _init_retriever(self, rules_path: str, faq_path: str, comments_path: str):
        with open(rules_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        texts = self.chunker.split_text(raw_text)
        with open(faq_path, "r", encoding="utf-8") as f:
            faq_strings_list = json.load(f)
        texts += faq_strings_list
        with open(comments_path, "r", encoding="utf-8") as f:
            comments = json.load(f)
        useful_comments = []
        for i in range(1, len(comments)):
            if comments[i - 1]["author"] == comments[i]["reply_to"]:
                useful_comments.append(comments[i - 1]["text"] + " " + comments[i]["text"])
        texts += useful_comments
        return ChromaRetrieverFactory().create_retriever(
            texts, self.embedder,
            search_kwargs={"k": 2},
            search_type="similarity"
        )

    def _init_prompt(self):
        from langchain_core.prompts import PromptTemplate
        template = """Ты - консультант по серии настольных игр "Эволюция"..."""
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def _init_rag_chain_from_docs(self):
        template = """Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

        Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.

        Используй только предоставленный контекст, чтобы кратко и ясно ответить на вопрос. Игнорируй не относящиеся к вопросу пользователя фрагменты контекста. Не повторяй фрагменты контекста дословно, если не нужно. Отвечай в одном-двух предложениях, если возможно.

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
                "question": lambda x: x[0].metadata.get("question", "") if x and hasattr(x[0], "metadata") else ""
            }
            | qa_prompt
            | self.generator
            | StrOutputParser()
        )

    def get_response(self, question: str) -> tuple[str, str]:
        retrieved_docs = self.retriever.invoke(question)
        for doc in retrieved_docs:
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata["question"] = question
        result = self.rag_chain_from_docs.invoke(retrieved_docs)
        truncated_result = result.split("\n")[0]
        last_dot_index = truncated_result.rfind(".")
        return truncated_result[:last_dot_index + 1].strip() if last_dot_index != -1 else truncated_result.strip(), RAGService._format_docs(retrieved_docs)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n".join([doc.page_content if hasattr(doc, "page_content") else doc["content"] for doc in docs])