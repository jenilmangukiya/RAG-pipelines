from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

model = ChatOpenAI(model="gpt-4o")

chat_history = []


def ask_question(question: str):
    final_question = ""
    if len(chat_history) == 0:
        final_question = question
    else:
        messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite the new question to ve standlone and searchable for RAG, just return the rewritten question as a string"
                )
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {question}")]
        )

        response = model.invoke(messages)
        final_question = response.content

    retriver = db.as_retriever(search_kwargs={"k": 3})
    docs = retriver.invoke(final_question)

    combined_input = f"""Based on the following document, please answer this quetstion {final_question}. Use the following context for answering: {docs}
    Please provide a clear, helpfull answr using only the information from these documents. if you can't find it answer i don't have enough information 
    """

    messages = [
        SystemMessage(
            content="You are a helpfull assistant that answrs questions based on provided documents and conversation"
        )
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    response = model.invoke(messages)
    answer = response.content

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    print("Answer: ", answer)
    return answer


def start_chat():
    print("Ask me question! Type 'quit' to exit.")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        ask_question(question)


start_chat()
