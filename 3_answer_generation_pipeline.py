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


combiled_input = f"""based on the following document, please answer this question: {query}

Documents:
{chr(10).join(f"- {doc.page_content}" for doc in relevant_docs)}

Please provide a clear, helpful answe using only the information from these documents. if you can't find the finswer in the documents, say "I don't have enough information to answer this question"
"""
model = ChatOpenAI(model="gpt-4o")

message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combiled_input),
]

response = model.invoke(message)

print("Answer:", response.content)
