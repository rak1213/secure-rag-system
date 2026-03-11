"""RAG Chain module using LangChain's LCEL (LangChain Expression Language)."""

from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from .retriever import format_docs_with_sources
from .logging_config import get_logger

log = get_logger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: list[str]
    source_documents: list[Document]

    def format_full_response(self) -> str:
        """Format the complete response with answer and sources."""
        response_parts = [self.answer]

        if self.sources:
            response_parts.append("\n--- Sources ---")
            for source in self.sources:
                response_parts.append(f"  {source}")

        return "\n".join(response_parts)


class RAGChain:
    """RAG chain using LangChain LCEL with model-agnostic LLM."""

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context from documents.

Instructions:
1. Answer the question using ONLY the information from the provided context.
2. If the context doesn't contain enough information to answer the question, clearly state: "I could not find information about this in the provided documents."
3. Be concise and direct in your answers.
4. Do not make up information or use knowledge outside the provided context.

Context:
{context}
"""

    def __init__(
        self,
        vector_store: Chroma,
        embeddings: Embeddings,
        llm: BaseChatModel,
        num_chunks: int = 4,
    ):
        """Initialize the RAG chain.

        Args:
            vector_store: ChromaDB vector store with indexed documents.
            embeddings: Embeddings model.
            llm: Language model instance.
            num_chunks: Number of chunks to retrieve for each query.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.num_chunks = num_chunks
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> RAGResponse:
        """Process a question through the full RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer and source citations.
        """
        log.info("rag.query.start", question=question[:80])

        # STEP 1: Embed the question
        query_vector = self.embeddings.embed_query(question)
        log.debug("rag.query.embedded", dimensions=len(query_vector))

        # STEP 2: Search the vector store
        docs = self.vector_store.similarity_search_by_vector(
            query_vector, k=self.num_chunks
        )

        if not docs:
            log.info("rag.query.no_results")
            return RAGResponse(
                answer="I could not find any relevant information in the provided documents.",
                sources=[],
                source_documents=[],
            )

        log.info("rag.query.retrieved", num_chunks=len(docs))

        # STEP 3: Format context and generate answer
        context, citations = format_docs_with_sources(docs)

        log.info("rag.query.generating", context_chars=len(context), num_chunks=len(docs))

        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        log.info("rag.query.done", answer_chars=len(answer))

        return RAGResponse(
            answer=answer,
            sources=citations,
            source_documents=docs,
        )

    def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """Process multiple questions."""
        return [self.query(q) for q in questions]
