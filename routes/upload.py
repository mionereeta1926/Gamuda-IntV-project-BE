import os

from fastapi import APIRouter, UploadFile, File

from services.file_loader import load_file_for_rag
from services.semantic_chunker import semantic_chunk_documents
from services.vectorstore import save_to_vectorstore

router = APIRouter()

UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    uploaded = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.lower().endswith((".pdf", ".docx", ".csv", ".xlsx")):
            docs = load_file_for_rag(file_path)
            chunks = semantic_chunk_documents(docs)
            save_to_vectorstore(chunks)
        else:
            raise ValueError("Unsupported file type. Only PDF, DOCX, CSV, and XLSX are allowed.")

        uploaded.append(file.filename)

    return {
        "uploaded_files": uploaded
    }