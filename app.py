from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.logging import router as logging_router
from routes.input_logging import router as input_logging_router

app = FastAPI(title="Project Intelligence Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(logging_router)
app.include_router(input_logging_router)

@app.get("/")
def health_check():
    return {
        "status": "running"
    }