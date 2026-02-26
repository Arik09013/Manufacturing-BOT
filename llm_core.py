# llm_core.py

# ----------------------------
# IMPORTS
# ----------------------------
import os
import re
import csv
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# ----------------------------
# LOAD API KEY
# ----------------------------
load_dotenv()  # Load variables from .env
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  # Pass key to OpenAI client

# ----------------------------
# CONFIG
# ----------------------------
pdf_folder = "E:/Env1/manufacturing_docs"
vector_db_dir = "E:/Env1/vector_db"
BATCH_SIZE = 500

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def auto_compute_numbers(text: str) -> str:
    """Automatically compute simple numeric expressions like '100 - 90'."""
    pattern = r'(\d+)\s*-\s*(\d+)'
    def repl(match):
        return str(int(match.group(1)) - int(match.group(2)))
    return re.sub(pattern, repl, text)

def display_answer(answer: str):
    """Clean, auto-compute, and display answer in Markdown."""
    from IPython.display import display, Markdown
    answer = re.sub(r"\\\[|\\\]", "", answer)
    answer = re.sub(r"\\text\{(.*?)\}", r"\1", answer)
    answer = answer.replace(" ,", ",").replace("  ", " ")
    answer = auto_compute_numbers(answer)
    display(Markdown(f"<div style='background:#f0f0f0;padding:10px;border-radius:8px'>{answer}</div>"))

# ----------------------------
# LOAD DOCUMENTS
# ----------------------------
docs = []

if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"Folder not found! Create: {pdf_folder}")
else:
    print(f"📁 PDF/TXT/CSV folder found: {pdf_folder}")

for file in os.listdir(pdf_folder):
    file_path = os.path.join(pdf_folder, file)

    # PDF
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # TXT
    elif file.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            docs.append(Document(page_content=text))

    # CSV
    elif file.endswith(".csv"):
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        csv_text = "\n".join(rows)
        docs.append(Document(page_content=csv_text))

print(f"✅ Total documents loaded: {len(docs)}")

# ----------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"📄 Total chunks created: {len(chunks)}")

# ----------------------------
# CREATE EMBEDDINGS & VECTOR DB
# ----------------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Chroma DB (empty if not exists)
db = Chroma(
    embedding_function=embeddings,
    persist_directory=vector_db_dir
)

# Add chunks in batches
for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i + BATCH_SIZE]
    db.add_documents(batch)

print("🚀 Vector database ready!")

# ----------------------------
# BOT FUNCTION
# ----------------------------
def ask_bot(question: str, history=None) -> str:
    """
    Answers manufacturing-related questions using OpenAI safely.
    Returns fixed response for unrelated questions.
    """
    # ----------------------------
    # 1. Detect manufacturing-related question
    # ----------------------------
    manufacturing_keywords = [
        "manufacturing", "machining", "metal cutting", "lathe", "milling",
        "process", "tolerance", "assembly", "industrial", "engineering",
        "CNC", "casting", "welding", "drilling"
    ]

    if not any(keyword.lower() in question.lower() for keyword in manufacturing_keywords):
        return "I can only answer manufacturing questions."

    # ----------------------------
    # 2. Get context from vector DB
    # ----------------------------
    results = db.similarity_search(question, k=3)
    if not results:
        return "Sorry, I could not find relevant information in your documents."

    context = "\n\n".join([doc.page_content for doc in results])

    # ----------------------------
    # 3. Combine conversation history
    # ----------------------------
    convo_text = ""
    if history:
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            convo_text += f"{role}: {content}\n"

    full_prompt = f"{convo_text}user: {question}\nassistant:"

    # ----------------------------
    # 4. Query OpenAI safely
    # ----------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in manufacturing & industrial measurement."},
                {"role": "user", "content": f"Context:\n{context}\n\nConversation:\n{full_prompt}"}
            ],
            max_tokens=300
        )
        answer = response.choices[0].message.content
        return auto_compute_numbers(answer)

    except Exception as e:
        # Handle quota exceeded or other errors
        err_msg = str(e)
        if "insufficient_quota" in err_msg or "RateLimitError" in err_msg:
            return "⚠️ API quota exceeded or too many requests. Please try again later."
        return f"⚠️ An error occurred: {err_msg}"