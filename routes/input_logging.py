from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import FileResponse

router = APIRouter()

@router.get("/inputsLogs/text")
def get_output_log():
    return FileResponse("logs/output_log.jsonl")