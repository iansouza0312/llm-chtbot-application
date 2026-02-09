from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List

def load_pdf_files(data):
  loader = DirectoryLoader(
    data,
    glob="*.pdf",
    loader_cls=PyPDFLoader
  )

  documents = loader.load()
  return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Filter a list of Document objects to only include the page content and source metadata."""
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs

# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunck = text_splitter.split_documents(minimal_docs)
    return text_chunck

def download_embeddings():
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceEmbeddings(model_name=model_name)
  return embeddings