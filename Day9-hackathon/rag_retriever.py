# rag_retriever.py

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

sources = [
    "docs/UX_guidelines.txt",
    "docs/brand_assets.txt",
    "docs/competitor_analysis.txt"
]

docs, meta = [], []
for src in sources:
    loader = TextLoader(src, encoding="utf-8")
    text = loader.load()
    docs.extend(text)
    meta.extend([{"source": src}] * len(text))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

def get_rag_context(query: str, k: int = 5):
    results = vector_store.similarity_search(query, k=k)
    return "\n---\n".join([f"From {doc.metadata['source']}:\n{doc.page_content}" for doc in results])
