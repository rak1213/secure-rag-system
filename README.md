# Secure RAG System

Production-ready Retrieval-Augmented Generation (RAG) system with guardrails, prompt injection defense, and evaluation metrics. Built for document Q&A over PDF files (demonstrated with Nova Scotia driving rules).

## Features

- **RAG Pipeline** — PDF ingestion, chunking, embedding, vector search, and LLM-based answer generation
- **Input Guardrails** — Query length validation, PII detection/redaction, off-topic filtering
- **Prompt Injection Defense** — 5-layer defense against adversarial inputs
- **Output Validation** — System prompt leak detection, response length caps
- **Evaluation Metrics** — Faithfulness scoring, retrieval relevance, refusal accuracy
- **Security Logging** — Dashboard tracking all security events

## Architecture

```
User Query
  |
  v
[INPUT GUARDRAILS] --> query length, PII detection, off-topic check
  |
  v
[PROMPT DEFENSES] --> injection detection, jailbreak refusal
  |
  v
[RETRIEVAL + CONFIDENCE CHECK] --> similarity search with relevance scores
  |
  v
[CONTEXT WRAPPING] --> instruction-data separation with delimiters
  |
  v
[LLM GENERATION] --> hardened system prompt + 30s timeout
  |
  v
[OUTPUT VALIDATION] --> leak check + response length cap
  |
  v
[EVALUATION] --> faithfulness scoring
  |
  v
Secure Response
```

## How to Run

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd secure-rag-system

# Install dependencies (requires uv)
uv sync

# Create .env file with your API keys
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY and JINA_API_KEY
```

### Usage

```bash
# Basic RAG system (interactive mode)
python main.py

# Basic RAG system (batch mode)
python main.py --batch

# Secure RAG system with all defenses + evaluation
python main_secure.py
```

Results are saved to `output/results.txt`.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Basic RAG entry point (interactive + batch modes) |
| `main_secure.py` | Secure RAG entry point with all defenses and test scenarios |
| `src/document_loader.py` | PDF document loading |
| `src/text_splitter.py` | Document chunking |
| `src/embeddings.py` | Jina AI embedding model |
| `src/vector_store.py` | ChromaDB vector store |
| `src/retriever.py` | Document retrieval and formatting |
| `src/rag_chain.py` | Basic RAG chain |
| `src/secure_rag_chain.py` | Enhanced RAG chain with all security layers |
| `src/guardrails.py` | Input/output guardrails and execution limits |
| `src/prompt_defense.py` | 5-layer prompt injection defense |
| `src/evaluation.py` | Faithfulness, retrieval relevance, and refusal accuracy metrics |
| `src/logger.py` | Security event logging dashboard |

## Prompt Injection Defenses (5 Layers)

1. **System Prompt Hardening** — Restricts the LLM to only answer driving questions, treats retrieved context as untrusted data, refuses to reveal instructions
2. **Input Sanitization** — Regex-based scanning for injection patterns: instruction overrides, role changes, prompt extraction, fake system markers, jailbreak keywords
3. **Instruction-Data Separation** — Retrieved chunks wrapped in XML tags; system prompt treats tag content as untrusted data
4. **Output Validation** — Post-generation scan for system prompt leaks or role-breaking indicators
5. **Jailbreak Refusal** — Intercepts jailbreak attempts before LLM and returns a fixed refusal message

## Evaluation Metrics

1. **Faithfulness Check** — LLM-based scoring of whether the answer is supported by retrieved context (0.0-1.0 scale)
2. **Retrieval Relevance** — Similarity scores for each retrieved chunk with confidence thresholds
3. **Refusal Accuracy** — Measures whether the system correctly answers valid questions and refuses invalid ones

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash (temperature=0.1)
- **Embeddings**: Jina AI v3 (1024 dimensions)
- **Vector Store**: ChromaDB (local, persistent)
- **Framework**: LangChain LCEL
- **Python**: 3.13+
