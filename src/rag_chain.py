"""RAG Chain module using LangChain's LCEL (LangChain Expression Language)."""

import os
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .retriever import create_retriever, format_docs_with_sources


@dataclass
class RAGResponse:
    """
    Response from the RAG system.

    Attributes:
        answer: The generated answer text.
        sources: List of source citations used.
        source_documents: The actual retrieved documents for reference.
    """
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
    """
    RAG (Retrieval Augmented Generation) chain using LangChain LCEL.

    Uses vector_store.similarity_search() for retrieval and
    prompt | llm | parser chain for generation.
    """

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
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        num_chunks: int = 4,
    ):
        """
        Initialize the RAG chain.

        Args:
            vector_store: ChromaDB vector store with indexed documents.
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
            model_name: Name of the Gemini model to use.
            num_chunks: Number of chunks to retrieve for each query.
        """
        self.vector_store = vector_store
        self.num_chunks = num_chunks

        # Initialize LLM
        google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.1,
        )

        # Create retriever using as_retriever()
        self.retriever = create_retriever(
            vector_store,
            search_type="similarity",
            k=num_chunks,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # Create LCEL chain: prompt | llm | parser
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> RAGResponse:
        """
        Process a question and generate an answer with sources.

        Uses the pattern from LangChain docs:
        1. Retrieve docs using similarity_search
        2. Format context with sources
        3. Generate answer using prompt | llm chain

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer and source citations.
        """
        # Retrieve relevant documents using similarity_search
        docs = self.vector_store.similarity_search(question, k=self.num_chunks)

        if not docs:
            return RAGResponse(
                answer="I could not find any relevant information in the provided documents.",
                sources=[],
                source_documents=[],
            )

        # Format context with source citations
        context, citations = format_docs_with_sources(docs)

        # Generate answer using LCEL chain
        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        return RAGResponse(
            answer=answer,
            sources=citations,
            source_documents=docs,
        )

    def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """
        Process multiple questions.

        Args:
            questions: List of questions to answer.

        Returns:
            List of RAGResponse objects.
        """
        return [self.query(q) for q in questions]
