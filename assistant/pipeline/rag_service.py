from typing import List

from assistant.components.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory
from assistant.components.retrievers.chroma_retriever_factory import ChromaRetrieverFactory
from assistant.components.generators.hf_model_generator import HFModelGenerator
from assistant.core.chunker import Chunker
from assistant.core.generator import Generator
from langchain_core.embeddings import Embeddings
import json

class RAGService:
    def __init__(self, chunker: Chunker = None, embedder: Embeddings = None, generator: Generator = None, use_rules: bool = True, use_faq: bool = True, use_comments: bool = True):
        self.chunker = chunker if chunker is not None else RecursiveCharacterChunker()
        self.embedder = embedder if embedder is not None else HFModelEmbedderFactory().create_embedder()
        self.retriever = self._init_retriever("data/documents/rules/knowledge-base-rules.txt",
                                              "data/documents/faq/faq.json",
                                              "data/documents/comments/comments.json",
                                              use_rules,
                                              use_faq,
                                              use_comments)
        self.generator = generator if generator is not None else HFModelGenerator()
        self.rag_chain_from_docs = self._init_rag_chain_from_docs()

    def _init_retriever(self, rules_path: str, faq_path: str, comments_path: str, use_rules: bool, use_faq: bool, use_comments: bool):
        assert use_rules or use_faq or use_comments, "At least one of sources should be used"
        texts = []
        if use_rules:
            with open(rules_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            texts += self.chunker.split_text(raw_text)
        if use_faq:
            with open(faq_path, "r", encoding="utf-8") as f:
                faq_strings_list = json.load(f)
            texts += faq_strings_list
        if use_comments:
            with open(comments_path, "r", encoding="utf-8") as f:
                comments = json.load(f)
            useful_comments = []
            for i in range(1, len(comments)):
                if comments[i - 1]["author"] == comments[i]["reply_to"]:
                    useful_comments.append(f"Пример пары вопрос-ответ:\nВопрос:\n{comments[i - 1]['text']}\nОтвет:\n{comments[i]['text']}")
            texts += useful_comments
        return ChromaRetrieverFactory().create_retriever(
            texts, self.embedder,
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.9},
            search_type="mmr"
        )

    def _init_rag_chain_from_docs(self):
        def rag_chain(retrieved_docs_for_each_question, prev_context_for_each_question):
            contexts = [RAGService._format_docs(retrieved_docs) for retrieved_docs in retrieved_docs_for_each_question]
            
            questions = []
            for retrieved_docs in retrieved_docs_for_each_question:
                if retrieved_docs and hasattr(retrieved_docs[0], "metadata"):
                    question = retrieved_docs[0].metadata.get("question", "")
                    questions.append(question)
                else:
                    questions.append("")
            
            system_content = """Ты - консультант по серии настольных игр "Эволюция". Ты должен помочь пользователю понять игровые правила, ответив на русском языке на его вопрос.

Суть игры заключается в том, чтобы создать наиболее жизнеспособную популяцию животных.
                    
Важно: ответь только на заданный вопрос. Не задавай встречных вопросов, не предлагай продолжить диалог и не генерируй новые темы. Для ответа на вопрос используй предоставленный контекст.

Важно: ты отвечаешь ТОЛЬКО на русском языке. Запрещено использовать китайские, английские или иные иностранные слова, символы или фразы. Все ответы должны содержать только кириллицу, цифры и базовые знаки препинания."""

            prompts = [[
                {"role": "system", "content": system_content},
                *prev_context,
                {"role": "user", "content": f"""
Контекст:
{context}

Вопрос:
{question}
"""}
            ] for context, question, prev_context in zip(contexts, questions, prev_context_for_each_question)]

            return self.generator(prompts)

        return rag_chain

    def get_response(self, questions: str | List[str], prev_context_for_each_question: str | List[str]) -> tuple[str, str]:
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
        
        answers = self.rag_chain_from_docs(retrieved_docs_for_each_question, prev_context_for_each_question)
        
        return answers, [RAGService._format_docs(docs) for docs in retrieved_docs_for_each_question]

    @staticmethod
    def _format_docs(docs) -> str:
        return [doc.page_content if hasattr(doc, "page_content") else doc["content"] for doc in docs]

    @staticmethod
    def format_docs(docs) -> str:
        return "\n\n".join(docs)
