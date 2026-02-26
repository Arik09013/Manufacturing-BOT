# batch_ingest_kaggle_fast_progress.py
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import csv
import torch

# ----------------------------
# Folders
# ----------------------------
os.environ["KAGGLE_CONFIG_DIR"] = r"E:\Env1\.kaggle"
download_folder = r"E:\Env1\manufacturing_docs"
vector_db_dir = r"E:\Env1\vector_db"

os.makedirs(download_folder, exist_ok=True)
os.makedirs(vector_db_dir, exist_ok=True)

# ----------------------------
# Kaggle API Authenticate
# ----------------------------
api = KaggleApi()
api.authenticate()

# ----------------------------
# Dataset list
# ----------------------------
datasets = [
    "ziya07/smart-manufacturing-iot-cloud-monitoring-dataset",
    "shasun/tool-wear-detection-in-cnc-mill",
]

# ----------------------------
# Download & Unzip Kaggle datasets
# ----------------------------
for ds in datasets:
    print(f"Downloading {ds} ...")
    api.dataset_download_files(ds, path=download_folder, unzip=True)
    print(f"✅ {ds} done!")

# ----------------------------
# Load all documents
# ----------------------------
docs = []
for file in os.listdir(download_folder):
    file_path = os.path.join(download_folder, file)
    try:
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(file_path).load())
        elif file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(Document(page_content=f.read()))
        elif file.endswith(".csv"):
            rows = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                rows = [", ".join(row) for row in reader]
            docs.append(Document(page_content="\n".join(rows)))
    except Exception as e:
        print(f"⚠️ Skipping {file} due to error: {e}")

print(f"✅ Total documents loaded: {len(docs)}")

# ----------------------------
# Split into chunks
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"📄 Total chunks created: {len(chunks)}")

# ----------------------------
# Fast GPU embeddings with progress
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
db = Chroma(embedding_function=embeddings, persist_directory=vector_db_dir)

BATCH_SIZE = 500  # adjust based on GPU memory
total_batches = len(chunks) // BATCH_SIZE + 1

print("🔵 Starting embedding loop with progress bar...")
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    # batch embedding (fast) and update vector DB
    db.add_documents(batch)
    db.persist()
    
    # Show progress
    print(f"➡️ Batch {i//BATCH_SIZE + 1}/{total_batches} processed ({len(batch)} chunks)")

print("✅ All chunks embedded and vector DB persisted")
