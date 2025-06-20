import time
import re
import json

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.core.config import hf_embeddings

# Load your JSON products data once
with open("app/db/grocery_products_50.json", "r") as f:
    json_data = json.load(f)
    all_products = []
    for section in json_data:
        all_products.extend(section.get("content", []))

# Enable caching
set_llm_cache(InMemoryCache())

# Load vectorstore
faiss_index = FAISS.load_local(
    'app/db/faiss_grocery_index_hf',
    hf_embeddings,
    allow_dangerous_deserialization=True
)

retriever = faiss_index.as_retriever(search_kwargs={"k": 3}, search_type='similarity')

POLITE_FALLBACK_MSG = (
    "I'm here to help with questions about our grocery store and products. "
    "Could you please ask something related to our shop?"
)

system_message = (
    "You are a helpful assistant for a grocery e-commerce shop in Bangladesh. "
    "Answer the user's questions ONLY using the information provided in the 'Context' section below. "
    "When the context includes user reviews, summarize the reviews concisely without revealing any usernames or exact comments. "
    "Use the conversation history to resolve pronouns, references, or ambiguous terms in the user's questions. "
    "If the question is unrelated to the shop's products, prices, stock, or shopping help, respond politely: "
    "\"I'm here to help with questions about our grocery store and products. Could you please ask something related to our shop?\"\n\n"
    "Do NOT use any external knowledge or make up answers. "
    "Always reference only the 'Context' section when responding."
)

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_template(
    """{system_message}

Context:
{context}

{chat_history}

User: {question}
Assistant:"""
).partial(system_message=system_message)

message_histories = {}

def get_history(session_id: str):
    if session_id not in message_histories:
        message_histories[session_id] = InMemoryChatMessageHistory()
    return message_histories[session_id]

# Store last product ID referenced per session
last_product_ids = {}

def save_last_product_id(session_id: str, product_id: str):
    last_product_ids[session_id] = product_id

def get_last_product_id(session_id: str):
    return last_product_ids.get(session_id)

chat_sessions = {}

def get_qa_chain_for_session(session_id: str):
    if session_id not in chat_sessions:
        chain = RunnableSequence(prompt | llm | StrOutputParser())
        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )
        chat_sessions[session_id] = {
            "chain": chain_with_history,
        }
    return chat_sessions[session_id]

def format_chat_history(history_messages):
    formatted = []
    for msg in history_messages:
        if msg.type == "human":
            formatted.append(f"User: {msg.content}")
        elif msg.type == "ai":
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

def count_products_by_keyword(keyword: str) -> int:
    all_docs = faiss_index.docstore._dict.values()
    keyword = keyword.lower()
    count = 0
    for doc in all_docs:
        metadata = doc.metadata
        if (
            keyword in metadata.get("category", "").lower() or
            keyword in metadata.get("sub_category", "").lower() or
            keyword in doc.page_content.lower()
        ):
            count += 1
    return count

def get_retriever_query(session_id: str, question: str) -> str:
    history = get_history(session_id).messages[-4:]  # last 4 messages
    formatted_history = format_chat_history(history)
    return f"{formatted_history}\nUser: {question}"

def get_answer_for_session(session_id: str, question: str) -> str:
    start_time = time.time()
    lower_q = question.lower()

    # 1. Detect product ID queries first
    match = re.search(r'\bP-\d{3}\b', question, re.IGNORECASE)
    if match:
        product_id = match.group().upper()
        product = next((p for p in all_products if p.get("product_id", "").upper() == product_id), None)
        if product:
            save_last_product_id(session_id, product_id)
            context_lines = []
            for key, value in product.items():
                if isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], dict):
                        # Pass only review ratings and comments without usernames
                        value = ", ".join(
                            f"Rating: {v['rating']}, Comment: {v['comment']}"
                            for v in value
                        )
                    else:
                        value = ", ".join(str(v) for v in value)
                context_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            context = "\n".join(context_lines)

            print(f"[DEBUG] Context for product ID {product_id}:\n{context}\n")

            session = get_qa_chain_for_session(session_id)
            chain = session["chain"]
            inputs = {
                "context": context,
                "question": question,
                "chat_history": format_chat_history(get_history(session_id).messages)
            }
            answer = chain.invoke(inputs, config={"configurable": {"session_id": session_id}})

            duration = time.time() - start_time
            print(f"[DEBUG] Answer generated in {duration:.2f} seconds: {answer}\n")
            return answer
        else:
            duration = time.time() - start_time
            print(f"[DEBUG] Product ID {product_id} not found. Responded in {duration:.2f} seconds.")
            return f"❌ Sorry, we couldn't find any product with ID **{product_id}**."

    # 2. Coreference / pronoun resolution: rewrite pronouns using last product ID
    pronouns = ["its", "their", "they", "them", "it"]
    if any(p in lower_q for p in pronouns):
        last_pid = get_last_product_id(session_id)
        if last_pid:
            question = re.sub(r'\bits\b', f'the product {last_pid}', question, flags=re.IGNORECASE)
            question = re.sub(r'\btheir\b', f'the product {last_pid}', question, flags=re.IGNORECASE)
            question = re.sub(r'\bthey\b', f'the product {last_pid}', question, flags=re.IGNORECASE)
            question = re.sub(r'\bthem\b', f'the product {last_pid}', question, flags=re.IGNORECASE)
            question = re.sub(r'\bit\b', f'the product {last_pid}', question, flags=re.IGNORECASE)
            lower_q = question.lower()
            print(f"[DEBUG] Question rewritten for coreference: {question}")

    # 3. Product count queries
    if "how many" in lower_q and "product" in lower_q:
        possible_keywords = ["snacks", "baby", "rice", "biscuits", "popcorn", "vegetables", "fruits"]
        for keyword in possible_keywords:
            if keyword in lower_q:
                count = count_products_by_keyword(keyword)
                duration = time.time() - start_time
                print(f"[DEBUG] Product count query responded in {duration:.2f} seconds.")
                return f"We currently have {count} {keyword} products in our grocery shop."
        total_products = len(faiss_index.index_to_docstore_id)
        duration = time.time() - start_time
        print(f"[DEBUG] Total product count query responded in {duration:.2f} seconds.")
        return f"We currently have {total_products} products in our grocery shop."

    # 4. Casual greetings
    casual_inputs = {
        "hi": "Hello! 👋 How can I assist you with your grocery shopping today?",
        "hello": "Hi there! 😊 Looking for something specific in our grocery shop?",
        "hey": "Hey! 👋 I'm here to help you find grocery items.",
        "are you available": "Yes, I'm always here to help with your grocery-related queries.",
        "how are you": "I'm great, thank you! How can I assist with your grocery needs?",
        "what is your name": "I'm your grocery assistant, here to help you with shopping!"
    }
    for key in casual_inputs:
        if key in lower_q:
            duration = time.time() - start_time
            print(f"[DEBUG] Casual response generated in {duration:.2f} seconds")
            return casual_inputs[key]

    # 5. RAG retrieval query with history-enhanced input
    session = get_qa_chain_for_session(session_id)
    chain = session["chain"]

    retriever_query = get_retriever_query(session_id, question)
    docs = retriever.invoke(retriever_query)

    if not docs:
        print(f"[DEBUG] No documents retrieved for question: {question}")
        return POLITE_FALLBACK_MSG

    context = "\n\n".join([doc.page_content.strip() for doc in docs if doc.page_content.strip()])
    if not context:
        print(f"[DEBUG] Retrieved docs, but context is empty for question: {question}")
        return POLITE_FALLBACK_MSG

    print(f"[DEBUG] Context for question '{question}':\n{context[:800]}...\n")

    inputs = {
        "context": context,
        "question": question,
        "chat_history": format_chat_history(get_history(session_id).messages)
    }

    answer = chain.invoke(inputs, config={"configurable": {"session_id": session_id}})
    end_time = time.time()

    print(f"[DEBUG] Answer generated in {end_time - start_time:.2f} seconds: {answer}")

    return answer
