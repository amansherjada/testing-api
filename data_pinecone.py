import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

# Set your Pinecone API key and environment
PINECONE_API_KEY = "pcsk_475ix6_QNMj2etqYWbrUz2aKFQebCPzCepmZEsZFoWsMG3wjYvFaxdUFu73h7GWbieTeti"

pc = Pinecone(PINECONE_API_KEY)
index = pc.Index("ahlchatbot-customer")

# Load data from the .txt file
file_path = 'MERGED_DATA.txt'
loader = TextLoader(file_path)
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=700, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# Create a Pinecone vector store
vector_store = PineconeVectorStore(index=index, embedding=embedding_model, text_key='text')

# Add documents to the vector store
vector_store.add_documents(documents= docs)

# Example similarity search
query = "Cancer wigs"
results = vector_store.similarity_search(query, k=2)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")