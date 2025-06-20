from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.config import hf_embeddings

# Enable LLM caching globally
set_llm_cache(InMemoryCache())

# Load FAISS vectorstore once
faiss_index = FAISS.load_local(
    'faiss_grocery_index_hf',
    hf_embeddings,
    allow_dangerous_deserialization=True
)

retriever = faiss_index.as_retriever(search_kwargs={"k": 3}, search_type='similarity')

# System prompt with instructions
system_message = SystemMessage(
    content=(
        "You are a helpful assistant for a grocery e-commerce shop in Bangladesh. "
        "Answer the user's questions ONLY using the information provided in the 'Context' section below. "
        "If the question is unrelated to the shop's products, prices, stock, or shopping help, respond politely: "
        "\"I'm here to help with questions about our grocery store and products. Could you please ask something related to our shop?\""
        "\n\n"
        "Do NOT use any external knowledge or make up answers. "
        "Always reference only the 'Context' section when responding."
    )
)

# Polite fallback if no relevant documents found
POLITE_FALLBACK_MSG = (
    "I'm here to help with questions about our grocery store and products. "
    "Could you please ask something related to our shop?"
)

# Groq Chat model instance
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    max_retries=2,
)

# Prepare prompt template including context and chat history
prompt = ChatPromptTemplate.from_messages([
    system_message,
    HumanMessage(content="Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{question}")
])

# Store active chat sessions (session_id => {chain, memory})
chat_sessions = {}

def get_qa_chain_for_session(session_id: str):
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        chain = RunnableSequence(
            prompt | llm | StrOutputParser()
        )

        chat_sessions[session_id] = {
            "chain": chain,
            "memory": memory
        }
    return chat_sessions[session_id]

def get_answer_for_session(session_id: str, question: str) -> str:
    session = get_qa_chain_for_session(session_id)
    chain = session["chain"]
    memory = session["memory"]

    # Add user question to conversation memory
    if hasattr(memory, "chat_memory"):
        memory.chat_memory.add_message(HumanMessage(content=question))

    # Retrieve relevant documents from FAISS vectorstore
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs if len(doc.page_content.strip()) > 0])

    # If no relevant documents, return polite fallback
    if not context.strip():
        return POLITE_FALLBACK_MSG

    # Run chain with question, context, and chat history
    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": memory.chat_memory.messages,
    })

    # Add assistant's answer to conversation memory
    if hasattr(memory, "chat_memory"):
        memory.chat_memory.add_message(AIMessage(content=answer))

    return answer
