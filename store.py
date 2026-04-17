from dotenv import load_dotenv
import os
from src.helper import load_pdf, filter_min_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
extracted_data = load_pdf(data="data/")
filter_data = filter_min_docs(extracted_data)
text_chunks = text_split(filter_data)
embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medchatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(index_name)

# Upload documents in batches to avoid 4MB size limit
batch_size = 50
docsearch = None

for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i : i + batch_size]
    print(
        f"Uploading batch {i // batch_size + 1}/{(len(text_chunks) - 1) // batch_size + 1}..."
    )

    if docsearch is None:
        # Create the vectorstore with the first batch
        docsearch = PineconeVectorStore.from_documents(
            documents=batch, embedding=embeddings, index_name=index_name
        )
    else:
        # Add subsequent batches
        docsearch.add_documents(batch)

print("✅ All documents uploaded to Pinecone!")
