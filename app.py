from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import *  # noqa: F403
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")

embeddings = download_embeddings()

index_name = "medchatbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings, index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.4,
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)


# Build RAG chain using modern langchain pattern
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User input: {msg}")
    response = rag_chain.invoke(msg)
    print("Response : ", response)
    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
