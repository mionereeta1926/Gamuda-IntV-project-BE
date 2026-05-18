import os
import zipfile
from xml.etree import ElementTree as ET

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from services.logging_service import log_print

DATAFRAME_STORE = {}

MISSING_VALUE_MARKERS = [
    "[not entered]",
    "not entered",
    "unassigned",
    "unknown",
    "#n/a",
    "n/a",
    "na",
    "none",
    "missing",
    "not available",
    "unspecified",
]


def normalize_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        normalized = df[col].astype("string").str.strip()
        normalized = normalized.replace({marker: pd.NA for marker in MISSING_VALUE_MARKERS})
        df[col] = normalized

    return df


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

    log_print(f"Loading spreadsheet file: {file_path}")

    source_name = os.path.basename(file_path)

    documents = []

    # CSV
    if file_path.lower().endswith(".csv"):

        # Read without header first, normalizing missing markers
        raw_df = pd.read_csv(
            file_path,
            header=None,
            na_values=MISSING_VALUE_MARKERS,
            keep_default_na=True,
        )

        # Find best header row
        header_row = detect_header_row(raw_df)

        # Reload correctly with cleaned missing values
        df = pd.read_csv(
            file_path,
            header=header_row,
            na_values=MISSING_VALUE_MARKERS,
            keep_default_na=True,
        )
        df = normalize_missing_data(df)

        DATAFRAME_STORE[file_path] = df

        documents.extend(
            _build_spreadsheet_documents(
                df,
                source_name,
                sheet_name="csv"
            )
        )

    # Excel
    else:

        excel_data = pd.read_excel(
            file_path,
            sheet_name=None,
            header=None,
            na_values=MISSING_VALUE_MARKERS,
            keep_default_na=True,
        )

        cleaned_sheets = {}

        for sheet_name, raw_df in excel_data.items():

            header_row = detect_header_row(raw_df)

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header_row,
                na_values=MISSING_VALUE_MARKERS,
                keep_default_na=True,
            )
            df = normalize_missing_data(df)

            cleaned_sheets[sheet_name] = df

            documents.extend(
                _build_spreadsheet_documents(
                    df,
                    source_name,
                    sheet_name
                )
            )

        DATAFRAME_STORE[file_path] = cleaned_sheets

    return documents


def detect_header_row(df, max_rows=10):
    """
    Detect which row most likely contains headers.
    """

    best_row = 0
    best_score = 0

    for i in range(min(max_rows, len(df))):

        row = df.iloc[i]

        non_null = row.notna().sum()

        string_count = sum(
            isinstance(x, str) and len(str(x).strip()) > 0
            for x in row
        )

        score = non_null + string_count

        if score > best_score:
            best_score = score
            best_row = i

    log_print(f"Detected header row: {best_row}")

    return best_row


def _build_spreadsheet_documents(df, source_name, sheet_name):

    documents = []

    for row_index, row in df.iterrows():

        row_items = []

        for column, value in row.items():

            if pd.isna(value):
                continue

            column = str(column).strip()
            value = str(value).strip()

            row_items.append(
                f"{column} - {value}"
            )

        if not row_items:
            continue

        content = (
            f"Row {row_index + 1}:\n" +
            "\n".join(row_items)
        )

        log_print(f"ROW CONTENT:\n{content}")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "sheet": sheet_name,
                    "row": int(row_index) + 1,
                    "page": int(row_index) + 1,
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