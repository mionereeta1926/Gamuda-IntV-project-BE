import os
import zipfile
from xml.etree import ElementTree as ET

import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

DATAFRAME_STORE = {}


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    basename = os.path.basename(file_path)

    for document in documents:
        document.metadata["source"] = basename

    return documents


def extract_text_from_docx(file_path):
    with zipfile.ZipFile(file_path) as docx_zip:
        document_xml = docx_zip.read("word/document.xml")

    root = ET.fromstring(document_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        if texts:
            paragraphs.append("".join(texts))

    return "\n".join(paragraphs)


def load_docx(file_path):
    content = extract_text_from_docx(file_path)
    return [
        Document(
            page_content=content,
            metadata={"source": os.path.basename(file_path), "page": 1},
        )
    ]


def load_csv_or_excel(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, sheet_name=None)

    DATAFRAME_STORE[file_path] = df

    documents = []
    source_name = os.path.basename(file_path)

    if file_path.endswith(".csv"):
        documents.extend(_build_spreadsheet_documents(df, source_name, sheet_name="csv"))
    else:
        for sheet_name, sheet_df in df.items():
            documents.extend(_build_spreadsheet_documents(sheet_df, source_name, sheet_name=sheet_name))

    return documents


def _build_spreadsheet_documents(df, source_name, sheet_name):
    documents = []

    for row_index, row in df.iterrows():
        row_items = []
        for column, value in row.items():
            if pd.isna(value):
                value_text = ""
            else:
                value_text = str(value)
            row_items.append(f"{column}: {value_text}")

        documents.append(
            Document(
                page_content="\n".join(row_items),
                metadata={
                    "source": source_name,
                    "sheet": sheet_name,
                    "row": int(row_index) + 1,
                },
            )
        )

    return documents


def load_file_for_rag(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".docx":
        return load_docx(file_path)
    if ext in {".csv", ".xlsx"}:
        return load_csv_or_excel(file_path)

    raise ValueError(f"Unsupported file type: {file_path}")

    DATAFRAME_STORE[file_path] = df

    return df