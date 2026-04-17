from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import torch
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        show_progress=True,
        silent_errors=True,
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def filter_min_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk


def download_embeddings():
    model_name = "pritamdeka/S-PubMedBert-MS-MARCO"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"✅ Embeddings loaded: {model_name}")
    return embeddings


embedding = download_embeddings()
