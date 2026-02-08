"""RAG Chain module using LangChain's LCEL (LangChain Expression Language)."""

import os
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from .retriever import format_docs_with_sources


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
        embeddings: Embeddings,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        num_chunks: int = 4,
    ):
        """
        Initialize the RAG chain.

        Args:
            vector_store: ChromaDB vector store with indexed documents.
            embeddings: Embeddings model (kept explicit so we can show each step).
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
            model_name: Name of the Gemini model to use.
            num_chunks: Number of chunks to retrieve for each query.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
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

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # Create LCEL chain: prompt | llm | parser
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> RAGResponse:
        """
        Process a question through the full RAG pipeline.

        Each step is made EXPLICIT so you can see:
          1. The question being converted into an embedding vector
          2. That vector being compared against stored document vectors
          3. The matched chunks being sent to the LLM for answer generation

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer and source citations.
        """
        # ── STEP 1: Embed the question ─────────────────────────────────
        # The SAME embedding model that was used for documents is now
        # used to convert the question into a vector. This puts the
        # question into the same vector space as the document chunks,
        # so we can compare them.
        print(f"\n┌─── QUERY STEP 1: EMBED THE QUESTION ─────────────────────────")
        print(f"│ Question: \"{question}\"")
        print(f"│ Sending to Jina AI to convert this text → vector...")

        query_vector = self.embeddings.embed_query(question)  # <-- embedding happens HERE

        print(f"│ Result: vector with {len(query_vector)} dimensions")
        print(f"│ Vector (first 8 values):")
        print(f"│   {[round(v, 4) for v in query_vector[:8]]}...")
        print(f"└───────────────────────────────────────────────────────────────\n")

        # ── STEP 2: Search the vector store ────────────────────────────
        # Now we take the question vector and find the closest document
        # vectors in ChromaDB using cosine similarity.
        # Note: we use similarity_search_by_vector() — NOT similarity_search()
        # — so the embedding step above stays visible and isn't hidden.
        print(f"┌─── QUERY STEP 2: SEARCH VECTOR STORE ────────────────────────")
        print(f"│ Comparing question vector against all {self.vector_store._collection.count()} stored document vectors...")
        print(f"│ Method: cosine similarity | Returning top {self.num_chunks} matches")
        print(f"│")

        docs = self.vector_store.similarity_search_by_vector(
            query_vector, k=self.num_chunks
        )

        if not docs:
            print(f"│ No matching documents found.")
            print(f"└───────────────────────────────────────────────────────────────\n")
            return RAGResponse(
                answer="I could not find any relevant information in the provided documents.",
                sources=[],
                source_documents=[],
            )

        print(f"│ Found {len(docs)} matching chunks:")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:70].replace("\n", " ")
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", 0) + 1
            print(f"│   Match {i}: \"{preview}...\"")
            print(f"│            (from {source}, page {page})")
        print(f"└───────────────────────────────────────────────────────────────\n")

        # ── STEP 3: Format context and generate answer ─────────────────
        # Take the matched chunks, format them as context, and send
        # everything (context + question) to Google Gemini to generate
        # a grounded answer.
        context, citations = format_docs_with_sources(docs)

        print(f"┌─── QUERY STEP 3: GENERATE ANSWER WITH LLM ──────────────────")
        print(f"│ Sending to Google Gemini:")
        print(f"│   - Context: {len(context)} chars from {len(docs)} retrieved chunks")
        print(f"│   - Question: \"{question}\"")
        print(f"│ Waiting for LLM response...")

        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        print(f"│ Answer received from Gemini ({len(answer)} chars)")
        print(f"└───────────────────────────────────────────────────────────────\n")

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
