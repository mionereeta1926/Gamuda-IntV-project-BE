import os

from fastapi import APIRouter, UploadFile, File, HTTPException

from services.file_loader import load_file_for_rag
from services.semantic_chunker import semantic_chunk_documents
from services.vectorstore import save_to_vectorstore
from services.logging_service import log_print
import time

router = APIRouter()

UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    start_time = time.time()
    log_print(f"Upload request received: {[f.filename for f in files]}")
    uploaded = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        log_print(f"Saved uploaded file to {file_path}")
        if file.filename.lower().endswith((".pdf", ".docx", ".csv", ".xlsx", ".xls")):
            docs = load_file_for_rag(file_path)
            chunks = semantic_chunk_documents(docs)
            save_to_vectorstore(chunks)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF, DOCX, CSV, and XLSX are allowed.",
            )

        uploaded.append(file.filename)
    time_taken_to_upload = (time.time() - start_time)
    log_print(f"Upload processing completed, Time taken: {time_taken_to_upload} seconds")
    return {
        "uploaded_files": uploaded
    }