from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

@router.get("/logs/text")
def read_log():
    with open("logs/print_log.log") as f:
        return {"content": f.read()}