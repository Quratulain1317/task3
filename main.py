#!/usr/bin/env python3
"""
Main CLI Interface for LangChain RAG System

This module provides a command-line interface for:
- Loading PDFs into the vector store
- Querying the RAG system with streaming support
- Interactive mode for multiple queries
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import config
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(directory_path: str) -> None:
    """
    Load PDF documents into the vector store.

    Args:
        directory_path: Path to directory containing PDFs
    """
    print("\n" + "=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)

    try:
        # Initialize components
        doc_loader = DocumentLoader()
        vector_store = VectorStore()

        # Check if directory exists
        if not Path(directory_path).exists():
            print(f"❌ Directory not found: {directory_path}")
            sys.exit(1)

        # Load and split documents
        print(f"\n📂 Loading PDFs from: {directory_path}")
        chunks = doc_loader.load_and_split(directory_path)

        if not chunks:
            print("❌ No documents found to process")
            sys.exit(1)

        # Get statistics
        stats = doc_loader.get_chunk_stats(chunks)
        print(f"\n📊 Document Statistics:")
        print(f"   Total chunks: {stats['count']}")
        print(f"   Average length: {stats['avg_length']:.0f} characters")
        print(f"   Min length: {stats['min_length']} characters")
        print(f"   Max length: {stats['max_length']} characters")

        # Create vector store
        print(f"\n🔄 Creating embeddings and storing in ChromaDB...")
        print(f"   This may take a few minutes...")
        vector_store.create_vectorstore(chunks)

        print(f"\n✅ Success! Documents indexed and ready for queries.")
        print(f"📁 Vector store saved to: {config.chroma_persist_directory}")
        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        sys.exit(1)


def query_rag_system(
        question: str,
        stream: bool = True,
        show_sources: bool = True
) -> None:
    """
    Query the RAG system and display results.

    Args:
        question: User question
        stream: Whether to use streaming mode
        show_sources: Whether to display source citations
    """
    print("\n" + "=" * 60)
    print("QUERYING RAG SYSTEM")
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    try:
        # Initialize components
        vector_store = VectorStore()

        # Load existing vector store
        print("\n📂 Loading vector store...")
        if vector_store.load_vectorstore() is None:
            print("❌ No vector store found. Please load documents first:")
            print("   python main.py --load data/papers/")
            sys.exit(1)

        # Create RAG chain
        print("🔗 Building RAG chain...")
        rag_chain = RAGChain(vector_store)

        # Get sources first (for display after streaming)
        sources = rag_chain.get_sources(question)

        # Generate answer
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)

        if stream:
            # Stream response token by token
            full_response = ""
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # New line after streaming
        else:
            # Get complete response at once
            response = rag_chain.invoke(question)
            print(response)

        # Display sources
        if show_sources and sources:
            print("\n" + "=" * 60)
            print("SOURCES:")
            print("=" * 60)

            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')

                # Clean up source path
                source_name = source.split('/')[-1] if '/' in source else source

                print(f"\n{i}. {source_name} (page {page})")

                # Show snippet
                snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(f"   Preview: {snippet}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)


def interactive_mode() -> None:
    """
    Start interactive query mode.

    Allows user to ask multiple questions without restarting.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE RAG MODE")
    print("=" * 60)
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    try:
        # Initialize components
        vector_store = VectorStore()

        # Load vector store
        print("\n📂 Loading vector store...")
        if vector_store.load_vectorstore() is None:
            print("❌ No vector store found. Please load documents first:")
            print("   python main.py --load data/papers/")
            sys.exit(1)

        # Create RAG chain
        print("🔗 Building RAG chain...")
        rag_chain = RAGChain(vector_store)
        print("✅ Ready for questions!\n")

        # Interactive loop
        while True:
            try:
                # Get question from user
                question = input("\n💬 Question: ").strip()

                # Check for exit commands
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if not question:
                    continue

                # Get sources
                sources = rag_chain.get_sources(question)

                # Stream answer
                print("\n🤖 Answer: ", end="", flush=True)
                for chunk in rag_chain.stream(question):
                    print(chunk, end="", flush=True)
                print()  # New line

                # Show sources
                if sources:
                    print(f"\n📚 Sources: {len(sources)} documents")
                    for i, doc in enumerate(sources, 1):
                        source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                        page = doc.metadata.get('page', 'N/A')
                        print(f"   {i}. {source_name} (page {page})")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Error in interactive mode: {e}")

    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="LangChain RAG System - Query research papers with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load documents
  python main.py --load data/papers/

  # Query with streaming (default)
  python main.py --query "What are transformers?"

  # Query without streaming
  python main.py --query "What are transformers?" --no-stream

  # Interactive mode
  python main.py --interactive

  # Hide sources
  python main.py --query "What are transformers?" --no-sources
        """
    )

    # ==========================================
    # 1. Primary Commands (Mutually Exclusive usually)
    # ==========================================

    # Argument: --load
    # Purpose: Ingest data. Expects a string value (directory path).
    # Usage: python main.py --load "data/papers"
    parser.add_argument(
        '--load',
        type=str,  # Treats input as a string
        metavar='PATH',  # Displayed in help menu as: --load PATH
        help='Load PDFs from directory into vector store'
    )

    # Argument: --query
    # Purpose: Ask one specific question. Expects a string value.
    # Usage: python main.py --query "What is X?"
    parser.add_argument(
        '--query',
        type=str,  # Treats input as a string
        metavar='QUESTION',  # Displayed in help menu as: --query QUESTION
        help='Query the RAG system'
    )

    # Argument: --interactive
    # Purpose: Enter chat mode. No value needed, just the flag.
    # Logic: action='store_true' means if present, set args.interactive = True.
    # Usage: python main.py --interactive
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive query mode'
    )

    # ==========================================
    # 2. Configuration Switches (Flags)
    # ==========================================

    # Argument: --stream (The "ON" switch)
    # Purpose: Explicitly enable streaming (redundant since default is True, but good for clarity).
    # Logic: Sets args.stream = True
    parser.add_argument(
        '--stream',
        action='store_true',
        default=True,  # <--- CRITICAL: This makes streaming ON by default!
        help='Enable streaming mode (default)'
    )

    # Argument: --no-stream (The "OFF" switch)
    # Purpose: Disable streaming if the user wants pure text output.
    # Logic: action='store_false' sets args.stream = False.
    # Note: dest='stream' links this flag to the SAME variable as above.
    # Usage: python main.py --query "Hi" --no-stream
    parser.add_argument(
        '--no-stream',
        action='store_false',
        dest='stream',  # Target the 'stream' variable defined above
        help='Disable streaming mode'
    )

    # Argument: --no-sources (The Opt-Out switch)
    # Purpose: Hide the list of source documents in the output.
    # Logic: Implicitly defaults to True (Show Sources). This flag flips it to False.
    # Usage: python main.py --query "Hi" --no-sources
    parser.add_argument(
        '--no-sources',
        action='store_false',  # Sets args.show_sources = False
        dest='show_sources',  # Creates variable args.show_sources
        help='Hide source citations'
    )

    args = parser.parse_args()

    # Validate Ollama connection
    print("\n🔍 Checking Ollama connection...")
    if not config.validate_ollama_connection():
        print("\n❌ Could not connect to Ollama.")
        print("Please make sure Ollama is running:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Pull required models:")
        print("      ollama pull llama3")
        print("      ollama pull nomic-embed-text")
        sys.exit(1)

    # Execute commands
    if args.load:
        load_documents(args.load)

    elif args.query:
        query_rag_system(
            question=args.query,
            stream=args.stream,
            show_sources=args.show_sources
        )

    elif args.interactive:
        interactive_mode()

    else:
        # No command specified
        parser.print_help()
        print("\n💡 Quick start:")
        print("   1. Load documents: python main.py --load data/papers/")
        print("   2. Query system: python main.py --query 'Your question here'")
        print("   3. Interactive mode: python main.py --interactive")


if __name__ == "__main__":
    main()