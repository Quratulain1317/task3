# LangChain RAG System with Streaming & LangSmith

A beginner-friendly Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, ChromaDB, and LangSmith for observability. Features real-time token-level streaming for enhanced user experience.

## Project Overview

This project demonstrates how to build a production-ready RAG system that:
- Processes research papers (PDFs) into searchable knowledge base
- Answers questions with context from your documents
- Provides source citations for transparency
- Streams responses token-by-token for immediate feedback
- Tracks all operations with LangSmith for debugging

## Learning Objectives

By exploring this project, you'll understand:

1. **RAG Pipeline Architecture**: How retrieval and generation work together
2. **Vector Embeddings**: Converting text to numerical representations for semantic search
3. **LangChain Expression Language (LCEL)**: Modern chain composition with `|` operator
4. **Token-Level Streaming**: Real-time response display for better UX
5. **LangSmith Tracing**: Observability into every step of your chain
6. **Document Chunking**: Breaking documents into retrievable segments

## Technology Stack

- **Python 3.10+**: Core programming language
- **LangChain**: Framework for LLM applications
- **Ollama**: Run LLMs locally (llama3 model)
- **ChromaDB**: Vector database for embeddings
- **LangSmith**: Observability and tracing platform
- **PyPDF**: PDF document processing

## Prerequisites

### 1. Install Ollama
```bash
# Visit https://ollama.ai and download for your OS
# Or use package manager:
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3
ollama run llama3 "What is the capital of France?" # For verification
ollama pull nomic-embed-text

# Verify installation
ollama --version
```

### 2. Python Environment
```bash
# Python 3.10 or higher required
python --version
```

### 3. LangSmith Account (Optional but Recommended)
1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Create a new project: "langchain-rag-learning"
3. Generate API key from settings
4. Keep handy for `.env` setup

## Installation

### Step 1: Clone/Download Project
```bash
# Create project directory
mkdir langchain-rag-project
cd langchain-rag-project
```

### Step 2: Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your LangSmith API key
# LANGCHAIN_API_KEY=your_api_key_here
```

### Step 5: Add Research Papers
```bash
# Place PDF files in data/papers/
mkdir -p data/papers
# Copy your PDF research papers into this folder
```

## Usage

### Loading Papers into Vector Store
```bash
# Index all papers in data/papers/
python main.py --load data/papers/

# Output:
# Loading PDFs from: data/papers/
# Loaded 3 documents
# Created 142 chunks
# ✅ Documents indexed successfully!
```

### Querying with Streaming (Default)
```bash
python main.py --query "What are transformers in deep learning?" --stream

# Output (appears token by token):
# 
# Answer: Transformers are a neural network architecture 
# introduced in the "Attention is All You Need" paper. They
# use self-attention mechanisms to process sequences...
# [text streams in real-time]
#
# Sources:
# - attention_is_all_you_need.pdf (page 3)
# - transformer_overview.pdf (page 1)
```

### Querying without Streaming
```bash
python main.py --query "What are transformers?" --no-stream

# Output (appears all at once):
# Answer: Transformers are...
# Sources: ...
```

### Interactive Mode
```bash
python main.py --interactive

# Starts interactive CLI:
# > What are the main benefits of transformers?
# Answer: [streaming response]...
# 
# > exit
```

## LangSmith Setup & Viewing Traces

### Enable Tracing
```bash
# In your .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=langchain-rag-learning
```

### View Traces
1. Run a query: `python main.py --query "test question"`
2. Visit [smith.langchain.com](https://smith.langchain.com)
3. Select project "langchain-rag-learning"
4. Click on latest trace to see:
   - Input question
   - Retrieved document chunks
   - LLM prompt with context
   - Generated response
   - Token usage & latency
   - Streaming behavior

### What You'll See in Traces
- **Retrieval Step**: Which chunks matched your query
- **Prompt Construction**: How context is formatted
- **LLM Generation**: Model output and timing
- **Token Streaming**: Individual chunk outputs
- **Errors**: Full stack traces if something fails

## Streaming Feature Explained

### Why Streaming Matters
Traditional LLM responses can take 5-10+ seconds. Without streaming, users see:
```
[waiting...]
[waiting...]
[complete response appears]
```

With streaming, users see:
```
Transformers are
Transformers are a
Transformers are a neural
Transformers are a neural network
...
```

### How It Works
1. **LCEL Native Support**: All LCEL chains support `.stream()`
2. **Token-Level Output**: Model generates one token at a time
3. **Immediate Display**: Tokens print as they arrive
4. **Better UX**: Users see progress, reducing perceived latency

### Implementation
```python
# In rag_chain.py, the chain is built with LCEL:
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm  # ChatOllama with streaming=True
    | StrOutputParser()
)

# Streaming is invoked with .stream():
for chunk in chain.stream(query):
    print(chunk, end="", flush=True)
```

## Project Structure

```
langchain-rag-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .env                         # Your config (don't commit!)
├── .gitignore                   # Git ignore rules
│
├── data/
│   ├── papers/                  # Place PDF papers here
│   └── chroma_db/               # ChromaDB storage (auto-created)
│
├── src/
│   ├── __init__.py              # Package marker
│   ├── config.py                # Configuration management
│   ├── document_loader.py       # PDF loading & chunking
│   ├── vector_store.py          # ChromaDB operations
│   ├── rag_chain.py             # Main RAG chain with LCEL
│   └── streaming.py             # Streaming utilities
│
├── notebooks/
│   └── demo.ipynb               # Interactive demo
│
└── main.py                      # CLI interface
```

### File Descriptions

- **config.py**: Loads environment variables, configures LangSmith
- **document_loader.py**: Uses PyPDFLoader and RecursiveCharacterTextSplitter
- **vector_store.py**: Manages ChromaDB for storing/retrieving embeddings
- **rag_chain.py**: Builds LCEL chain for retrieval + generation
- **streaming.py**: Handles token-level streaming and formatting
- **main.py**: Command-line interface with argparse

## Why Use LangChain?

You could build RAG systems without LangChain. Here's what LangChain provides:

**Key Benefits:**
- **LCEL Streaming**: Built-in token streaming with `.stream()` - no manual callback handling
- **LangSmith Tracing**: Automatic observability for every chain step
- **Swappable Components**: Change vector stores, LLMs, or embeddings with minimal code changes
- **100+ Integrations**: Pre-built connectors for popular tools (ChromaDB, Pinecone, OpenAI, etc.)

**When to use raw code instead:**
- Simple single LLM API calls
- Highly custom logic where abstractions add complexity
- Learning fundamentals from scratch

**Bottom line:** LangChain saves you from building streaming infrastructure, observability, and integrations. For basic RAG, you could code it yourself in ~200 lines, but LangChain handles edge cases and provides production patterns.

## 🧠 Key Concepts

### What is RAG?
**Retrieval-Augmented Generation** combines:
1. **Retrieval**: Find relevant documents from your knowledge base
2. **Augmentation**: Add retrieved context to LLM prompt
3. **Generation**: LLM generates answer using context

**Without RAG**: LLM only knows training data
**With RAG**: LLM can answer using your specific documents

### Vector Embeddings
Text converted to numerical vectors that capture semantic meaning:
```
"cat" → [0.2, 0.8, 0.1, ...]
"kitten" → [0.3, 0.7, 0.2, ...]  # Similar to "cat"
"car" → [0.9, 0.1, 0.8, ...]     # Different from "cat"
```

Similar concepts have similar vectors, enabling semantic search.

### Vector Stores (ChromaDB)
Databases optimized for:
- Storing high-dimensional vectors
- Fast similarity search
- Retrieving most relevant chunks

### Document Chunking
Breaking documents into smaller pieces because:
- LLMs have context limits
- Smaller chunks = more precise retrieval
- Balance: too small loses context, too large is imprecise

### LCEL (LangChain Expression Language)
Modern way to build chains with `|` operator:
```python
chain = component1 | component2 | component3
result = chain.invoke(input)
```

Benefits:
- Declarative syntax
- Built-in streaming
- Automatic LangSmith tracing
- Easy to customize

### Token-Level Streaming
Models generate text one token (~word) at a time. Streaming displays each token immediately instead of waiting for completion.

## Troubleshooting

### Issue: Ollama Connection Error
```
Error: Could not connect to Ollama at http://localhost:11434
```
**Solution**:
```bash
# Start Ollama service
ollama serve

# In another terminal, verify:
curl http://localhost:11434
```

### Issue: Model Not Found
```
Error: model "llama3" not found
```
**Solution**:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Issue: No Documents Found
```
Warning: No PDF files found in data/papers/
```
**Solution**:
```bash
# Add PDFs to the directory
cp /path/to/your/papers/*.pdf data/papers/
```

### Issue: LangSmith Not Tracing
```
Warning: LangSmith tracing disabled
```
**Solution**:
```bash
# Check .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_actual_key
LANGCHAIN_PROJECT=langchain-rag-learning
```

### Issue: ChromaDB Errors
```
Error: ChromaDB initialization failed
```
**Solution**:
```bash
# Clear and rebuild database
rm -rf data/chroma_db/
python main.py --load data/papers/
```

### Issue: Slow Responses
**Streaming helps perceived performance**, but if truly slow:
- Check CPU usage (Ollama is CPU-intensive)
- Try smaller model: `ollama pull llama3:8b-instruct-q4_0`
- Reduce chunk size in `document_loader.py`

### Issue: Poor Answer Quality
- **Add more context**: Increase chunks retrieved (k parameter)
- **Better chunking**: Adjust chunk_size and overlap
- **Prompt engineering**: Modify prompt in `rag_chain.py`
- **Better embeddings**: Try different embedding models

### Resources
- [LangChain Docs](https://docs.langchain.com/oss/python/langchain/overview)
- [LangSmith Guide](https://docs.langchain.com/langsmith/home)
- [ChromaDB Docs](https://docs.trychroma.com/docs/overview/introduction)
- [Ollama Models](https://ollama.ai/library)

## Contributing

This is a learning project! Feel free to:
- Fork and experiment
- Share improvements
- Ask questions in issues
- Submit PRs with enhancements

## License

MIT License - Free to use for learning and commercial projects.
---

**Happy Learning!**

Questions? Open an issue or check the troubleshooting guide above.