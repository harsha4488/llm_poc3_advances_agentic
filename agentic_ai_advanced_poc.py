"""
ADVANCED AGENTIC AI PoC
- Multi-agent (Planner, Executor, Critic)
- Memory across turns
- Tool usage
- Self-critique & retry
- Human escalation
Python 3.11 recommended
"""

from typing import List, Dict
from pathlib import Path
import re

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"
VECTOR_DIR = "vector_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CRITIC_THRESHOLD = 0.6
MAX_RETRIES = 2


# -----------------------------
# Tools
# -----------------------------
def tool_order_status(order_id: str) -> str:
    return f"Order {order_id} is shipped and will arrive in 2 days."


def tool_refund_policy() -> str:
    return "Refunds are processed within 5 business days."


# -----------------------------
# Memory Store
# -----------------------------
class Memory:
    def __init__(self):
        self.history: List[Dict] = []

    def add(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})

    def summarize(self) -> str:
        return " ".join(
            f"Q:{h['question']} A:{h['answer']}" for h in self.history[-3:]
        )


# -----------------------------
# Vector Store
# -----------------------------
def ingest_documents():
    docs = []
    for file in Path(DATA_DIR).glob("*.txt"):
        docs.extend(TextLoader(str(file)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)


def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        VECTOR_DIR, embeddings, allow_dangerous_deserialization=True
    )


# -----------------------------
# PLANNER AGENT
# -----------------------------
def planner_agent(question: str) -> str:
    q = question.lower()

    if "order" in q and re.search(r"\d+", q):
        return "use_tool_order"

    if "refund" in q:
        return "use_tool_refund"

    return "use_retrieval"


# -----------------------------
# EXECUTOR AGENT
# -----------------------------
def executor_agent(plan: str, question: str, db) -> str:
    if plan == "use_tool_order":
        order_id = re.search(r"\d+", question).group()
        return tool_order_status(order_id)

    if plan == "use_tool_refund":
        return tool_refund_policy()

    docs = db.similarity_search(question, k=3)
    return " ".join(d.page_content for d in docs)


# -----------------------------
# CRITIC AGENT
# -----------------------------
def critic_agent(answer: str, context: str) -> float:
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    if not answer_tokens:
        return 0.0

    return len(answer_tokens & context_tokens) / len(answer_tokens)


# -----------------------------
# MAIN LOOP
# -----------------------------
def run():
    if not Path(VECTOR_DIR).exists():
        ingest_documents()

    db = load_db()
    memory = Memory()

    print("\nADVANCED AGENTIC AI SYSTEM")
    print("Type 'exit' to quit\n")

    while True:
        question = input("❓ User: ")
        if question.lower() == "exit":
            break

        retries = 0
        context = memory.summarize()
        final_answer = None

        while retries <= MAX_RETRIES:
            plan = planner_agent(question)
            answer = executor_agent(plan, question, db)

            score = critic_agent(answer, context + answer)

            if score >= CRITIC_THRESHOLD:
                final_answer = answer
                break

            retries += 1

        if final_answer is None:
            print("Escalated to human agent (low confidence)")
            continue

        memory.add(question, final_answer)

        print(f"\nPlan: {plan}")
        print(f"Confidence: {round(score, 2)}")
        print(f"Answer: {final_answer}\n")


if __name__ == "__main__":
    run()