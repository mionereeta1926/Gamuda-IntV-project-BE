# FastAPI Setup
## 1) Create and activate virtual environment
Windows (PowerShell / CMD):

bash
venv\Scripts\activate

## 2) Install dependencies
bash
pip install -r requirements.txt

## 3) Run the FastAPI app
bash
uvicorn app:app --reload

## 4) Open in your browser
The server will start at the local URL shown in the terminal (commonly http://127.0.0.1:8000). Open /docs for Swagger UI.
