import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):

    # load all txt files from folder specified
    loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in {docs_path}")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f" Source: {doc.metadata["source"]}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Page content: {doc.page_content[:100]}...")
        print(f" Metadata : {doc.metadata}")
        print("-" * 50)

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    # Split documents into chunks
    print("Splitting documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nSplit {len(chunks)} chunks.")
    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1}")
            print(f" Content length: {len(chunk.page_content)} characters")
            print(f" Page content: {chunk.page_content}...")
            print(f" Metadata : {chunk.metadata}")
            print("-" * 50)

    return chunks


def crate_vector_store(chunks, persist_directory="db/chroma_db"):
    # Create and persist ChromaDB vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store created at {persist_directory}")
    return vectorstore


def main():
    # Load documents
    documents = load_documents()

    # Split documents into chunks
    chunks = split_documents(documents)

    # Embedding documents and creating vector store
    vectorstore = crate_vector_store(chunks)

    print("Vector store created")


main()
