import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pathlib import Path


# Embedding function
_embedder = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ChromaDB client 
_client = chromadb.PersistentClient(path=".chroma")


class RAGPipeline:
    """
    Ingests documents into ChromaDB, retrieves relevant chunks,
    and generates answers using any BaseLLMAdapter.
    """

    def __init__(self, collection_name: str = "eval_docs"):
        self.collection = _client.get_or_create_collection(
            name=collection_name,
            embedding_function=_embedder,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    # Ingestion 
    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Chunk and embed a raw string. Returns number of chunks added."""
        chunks = self.splitter.split_text(text)
        ids = [f"{source}::{i}" for i in range(len(chunks))]

        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": source}] * len(chunks),
        )
        return len(chunks)

    def ingest_pdf(self, pdf_path: str) -> int:
        """Extract text from a PDF and ingest it."""
        reader = PdfReader(pdf_path)
        full_text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
        source = Path(pdf_path).name
        return self.ingest_text(full_text, source=source)

    def ingest_file(self, file_path: str) -> int:
        """Auto-detect txt or pdf and ingest."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self.ingest_pdf(file_path)
        else:
            return self.ingest_text(path.read_text(), source=path.name)

    # Retrieval
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k most relevant chunks for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )
        return results["documents"][0]  # list of chunk strings

    # Generate (RAG answer)
    def answer(self, question: str, adapter, top_k: int = 3) -> tuple[str, str]:
        """
        Retrieve context, then generate an answer.
        Returns (answer, retrieved_context) so the eval harness
        can score against the actual retrieved context.
        """
        chunks = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(chunks)

        response = adapter.complete(
            system_prompt="Answer using only the provided context. Be concise.",
            user_prompt=f"Context:\n{context}\n\nQuestion:\n{question}",
        )
        return response, context