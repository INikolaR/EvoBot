from typing import List

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
            search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.9},
            search_type="mmr"
        )

    def _init_rag_chain_from_docs(self):
        def format_conversation(retrieved_docs_for_each_question):

            contexts = [RAGService._format_docs(retrieved_docs) for retrieved_docs in retrieved_docs_for_each_question]
            
            questions = []
            for retrieved_docs in retrieved_docs_for_each_question:
                if retrieved_docs and hasattr(retrieved_docs[0], "metadata"):
                    question = retrieved_docs[0].metadata.get("question", "")
                    questions.append(question)
                else:
                    questions.append("")
            
            system_content = """Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.
                    
Важно: ответь только на заданный вопрос. Не задавай встречных вопросов, не предлагай продолжить диалог и не генерируй новые темы. Для ответа на вопрос используй предоставленный контекст."""

            return [[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"""
Контекст:
{context}

Вопрос:
{question}
"""}
            ] for context, question in zip(contexts, questions)]

        return format_conversation | self.generator

    def get_response(self, questions: str | List[str]) -> tuple[str, str]:
        if isinstance(questions, str):
            questions = [questions]
        retrieved_docs_for_each_question = []
        for question in questions:
            retrieved_docs = self.retriever.invoke(question)
            for doc in retrieved_docs:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata["question"] = question
            retrieved_docs_for_each_question.append(retrieved_docs)
        
        answers = self.rag_chain_from_docs.invoke(retrieved_docs_for_each_question)

        # truncated_result = result.split("\n")[0]
        # last_dot_index = truncated_result.rfind(".")
        # final_answer = truncated_result[:last_dot_index + 1].strip() if last_dot_index != -1 else truncated_result.strip()
        
        return answers, RAGService._format_docs(retrieved_docs_for_each_question)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n".join([doc.page_content if hasattr(doc, "page_content") else doc["content"] for doc in docs])
