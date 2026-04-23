from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def get_retriever():
    embeddings = OpenAIEmbeddings()

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return db.as_retriever(search_kwargs={"k": 3})

from langchain.chat_models import ChatOpenAI

def generate_answer(query, docs):
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a customer support assistant.

    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    llm = ChatOpenAI(temperature=0)

    response = llm.predict(prompt)

    return response