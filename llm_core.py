# ----------------------------
# IMPORTS
# ----------------------------
import os
import re
import csv
import json
from dotenv import load_dotenv
from tqdm import tqdm

from google import genai

from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# ----------------------------
# LOAD API KEY
# ----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

# ----------------------------
# CONFIG
# ----------------------------
pdf_folder = "E:/Ai/manufacturing_docs"
vector_db_dir = "E:/Ai/vector_db"
BATCH_SIZE = 500

# ----------------------------
# HELPER
# ----------------------------
def auto_compute_numbers(text: str) -> str:
    return re.sub(r'(\d+)\s*-\s*(\d+)', lambda m: str(int(m.group(1)) - int(m.group(2))), text)

# ----------------------------
# LOAD DOCS
# ----------------------------
docs = []

if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"{pdf_folder} not found")

print("📁 Loading documents...")

for file in os.listdir(pdf_folder):
    path = os.path.join(pdf_folder, file)

    if file.endswith(".pdf"):
        docs.extend(PyPDFLoader(path).load())

    elif file.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(Document(page_content=f.read()))

    elif file.endswith(".csv"):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f):
                rows.append(", ".join(row))
        docs.append(Document(page_content="\n".join(rows)))

print(f"✅ Loaded: {len(docs)} docs")

# ----------------------------
# SPLIT
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

print(f"📄 Chunks: {len(chunks)}")

# ----------------------------
# VECTOR DB
# ----------------------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    embedding_function=embeddings,
    persist_directory=vector_db_dir
)

if db._collection.count() == 0:
    print("⚡ Creating vector DB...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        db.add_documents(chunks[i:i+BATCH_SIZE])
    db.persist()
else:
    print("✅ Vector DB already exists")

# ----------------------------
# GREETING
# ----------------------------
def is_greeting(text):
    patterns = [
        r"\b(hi|hello|hey|yo|salam|হ্যালো|আসসালামুয়ালাইকুম)\b",
        r"\b(how are you|what's up|sup)\b"
    ]
    return any(re.search(p, text.lower()) for p in patterns)

# ----------------------------
# CLASSIFY (GEMINI)
# ----------------------------
def classify_topic(question):
    try:
        prompt = f"""
Return ONLY JSON:

{{"language": "", "corrected": "", "is_manufacturing": true/false}}

Rules:
- Detect language
- Fix grammar
- Manufacturing topics only

Input:
{question}
"""

        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        raw = res.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        return json.loads(raw)

    except:
        return {"language": "English", "corrected": question, "is_manufacturing": False}

# ----------------------------
# MAIN BOT
# ----------------------------
def ask_bot(question, history=None):

    # Greeting
    if is_greeting(question):
        prompt = f"""
User greeted you.

Reply friendly, short, same language.
Introduce yourself as Manufacturing AI.

User: {question}
"""
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return res.text

    # Classification
    cls = classify_topic(question)
    lang = cls["language"]
    corrected = cls["corrected"]
    is_manu = cls["is_manufacturing"]

    # Reject
    if not is_manu:
        prompt = f"""
User asked non-manufacturing question.

Politely refuse in {lang} in 1-2 lines.

User: {question}
"""
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return res.text

    # Retrieve context
    results = db.similarity_search(corrected, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # History
    history_text = ""
    if history:
        for msg in history:
            history_text += f"{msg['role']}: {msg['content']}\n"

    # Final prompt
    prompt = f"""
You are a Manufacturing AI expert.

Language: {lang}

Context:
{context}

Conversation:
{history_text}

User: {corrected}

Give clear structured answer.
"""

    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    answer = res.text

    if corrected.lower().strip() != question.lower().strip():
        answer = f'(Auto-corrected: "{corrected}")\n\n' + answer

    return auto_compute_numbers(answer)