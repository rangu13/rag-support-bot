from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def ingest_pdf():
    loader = PyPDFLoader("data/knowledge.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )

    db.persist()
    print("✅ PDF processed and stored in ChromaDB")