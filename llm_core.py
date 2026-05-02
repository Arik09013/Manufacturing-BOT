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
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ----------------------------
# CONFIG
# ----------------------------
pdf_folder = "E:/Ai/manufacturing_docs"
vector_db_dir = "E:/Ai/vector_db"
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
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    elif file.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            docs.append(Document(page_content=text))
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

db = Chroma(
    embedding_function=embeddings,
    persist_directory=vector_db_dir
)

if db._collection.count() == 0:
    print("Creating vector database for first time...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i:i + BATCH_SIZE]
        db.add_documents(batch)
    db.persist()
    print("Vector DB created successfully!")
else:
    print("Vector DB already exists. Skipping embedding.")

# ----------------------------
# GREETING DETECTION
# ----------------------------
GREETING_PATTERNS = [
    r"\b(hi|hello|hey|hiya|howdy|greetings|good\s*(morning|afternoon|evening|night)|what'?s\s*up|sup|yo)\b",
    r"\b(salut|bonjour|bonsoir|ciao|hola|namaste|salam|নমস্কার|হ্যালো|আসসালামুয়ালাইকুম)\b",
    r"\b(how are you|how r u|how do you do|how's it going|how are things)\b",
    r"\b(nice to meet|good to meet|pleased to meet)\b",
    r"^\s*(ok|okay|alright|sure|thanks|thank you|thx|ty)\s*[!.?]*\s*$",
]

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting or small-talk."""
    lower = text.lower().strip()
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False

# ----------------------------
# TOPIC CLASSIFICATION VIA LLM
# ----------------------------
def classify_topic(question: str) -> dict:
    """
    Use GPT to:
    1. Detect the language of the question
    2. Correct grammar
    3. Classify if it's manufacturing-related
    Returns dict: {language, corrected_question, is_manufacturing}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a smart language analyzer. Given a user message, respond ONLY in this JSON format:\n"
                        '{"language": "<detected language name>", "corrected": "<grammar-corrected version in same language>", "is_manufacturing": <true|false>}\n\n'
                        "Rules:\n"
                        "- 'is_manufacturing' is true ONLY if the question is about: manufacturing, machining, CNC, welding, casting, milling, drilling, lathe, tolerances, assembly, industrial processes, metal cutting, quality control, engineering measurements.\n"
                        "- 'is_manufacturing' is false for: general knowledge, coding, math, health, cooking, sports, entertainment, etc.\n"
                        "- Correct grammar/spelling mistakes in 'corrected' field but keep the same language.\n"
                        "- Do NOT add any explanation or extra text. Only pure JSON."
                    )
                },
                {"role": "user", "content": question}
            ],
            max_tokens=150
        )
        import json
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {"language": "English", "corrected": question, "is_manufacturing": False}

# ----------------------------
# BOT FUNCTION
# ----------------------------
def ask_bot(question: str, history=None) -> str:
    """
    Answers manufacturing-related questions using OpenAI.
    - Handles greetings naturally (like ChatGPT)
    - Auto-corrects grammar
    - Detects and responds in the user's language
    - Refuses non-manufacturing questions politely
    """

    # ── 1. Handle greetings / small talk ──────────────────────────────────────
    if is_greeting(question):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly Manufacturing AI Assistant. "
                            "The user has greeted you or is making small talk. "
                            "Respond warmly and naturally in the SAME language the user used. "
                            "Introduce yourself briefly as a Manufacturing AI Assistant "
                            "and invite them to ask their manufacturing question. "
                            "Keep it short, friendly, and conversational — like ChatGPT."
                        )
                    },
                    {"role": "user", "content": question}
                ],
                max_tokens=120
            )
            return response.choices[0].message.content
        except Exception as e:
            return "Hello! 👋 I'm your Manufacturing AI Assistant. How can I help you today?"

    # ── 2. Classify topic, detect language, correct grammar ───────────────────
    classification = classify_topic(question)
    language = classification.get("language", "English")
    corrected_question = classification.get("corrected", question)
    is_manufacturing = classification.get("is_manufacturing", False)

    # ── 3. Reject non-manufacturing questions ─────────────────────────────────
    if not is_manufacturing:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a Manufacturing AI Assistant that ONLY answers manufacturing-related questions. "
                            f"The user asked something unrelated to manufacturing. "
                            f"Politely decline in {language} and remind them you can only help with manufacturing topics. "
                            f"Be friendly, not rude. Keep it 1-2 sentences."
                        )
                    },
                    {"role": "user", "content": question}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception:
            return "I'm specialized in manufacturing topics only. Please ask me about machining, CNC, welding, casting, or other manufacturing processes! 🏭"

    # ── 4. Get relevant context from vector DB ────────────────────────────────
    results = db.similarity_search(corrected_question, k=3)
    context = "\n\n".join([doc.page_content for doc in results]) if results else ""

    # ── 5. Build conversation history ─────────────────────────────────────────
    messages_payload = [
        {
            "role": "system",
            "content": (
                f"You are an expert Manufacturing AI Assistant specializing in industrial processes, "
                f"machining, CNC, welding, casting, quality control, and engineering measurements. "
                f"IMPORTANT: Always respond in {language} — the same language the user is using. "
                f"Be clear, accurate, and professional. Use bullet points or numbered lists when helpful. "
                f"If the context below is relevant, use it. Otherwise use your expert knowledge.\n\n"
                f"Context from documents:\n{context}"
            )
        }
    ]

    # Add conversation history
    if history:
        for msg in history[:-1]:  # Exclude the last message (current question)
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages_payload.append({"role": role, "content": content})

    # Add the corrected current question
    messages_payload.append({"role": "user", "content": corrected_question})

    # ── 6. Query OpenAI ───────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_payload,
            max_tokens=500
        )
        answer = response.choices[0].message.content

        # Show grammar correction note if question was corrected
        if corrected_question.strip().lower() != question.strip().lower():
            answer = f"*(Auto-corrected: \"{corrected_question}\")*\n\n{answer}"

        return auto_compute_numbers(answer)

    except Exception as e:
        err_msg = str(e)
        if "insufficient_quota" in err_msg or "RateLimitError" in err_msg:
            return "⚠️ API quota exceeded or too many requests. Please try again later."
        return f"⚠️ An error occurred: {err_msg}"
