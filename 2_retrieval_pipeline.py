from langchain_community.document_loaders import bigquery
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory = "db/chroma_db"

embeding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


query = "what is nvidia doing?"

retriver = db.as_retriever(search_kwargs={"k": 3})


relevant_docs = retriver.invoke(query)

print(f"User query : {query}")
print("-" * 50)

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n {doc.page_content}\n")
