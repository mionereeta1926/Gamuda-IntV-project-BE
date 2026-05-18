import re
from langchain_text_splitters import TokenTextSplitter
from services.logging_service import log_print

# Exact unwanted values
STRICT_EXCLUDE = {
    'nan',
    'null',
    'none',
    '#n/a',
    'unknown'
}

# Regex patterns for noisy text
NOISE_PATTERNS = [
    r'unnamed:\s*\d*',
    r'column\d+',
    r'page\s*\d+',
    r'cid:\d+',
    r'#\w+!\??'
]


def clean_text(text):
    """
    Removes only unwanted words/patterns,
    NOT the entire line.
    """

    if not isinstance(text, str):
        return ""

    cleaned = text

    # Remove exact unwanted standalone words
    for word in STRICT_EXCLUDE:
        cleaned = re.sub(
            rf'\b{re.escape(word)}\b',
            '',
            cleaned,
            flags=re.IGNORECASE
        )

    # Remove noise patterns only
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(
            pattern,
            '',
            cleaned,
            flags=re.IGNORECASE
        )

    # Normalize spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


def clean_documents(documents):
    cleaned_documents = []

    for i, doc in enumerate(documents):

        if not hasattr(doc, "page_content"):
            continue

        content = doc.page_content

        if not isinstance(content, str):
            continue

        log_print(f"RAW DOC: {i}")
        log_print(content[:2000])

        # Clean only unwanted words/patterns
        cleaned_content = clean_text(content)

        log_print(f"CLEANED DOC: {i}")
        log_print(cleaned_content[:2000])

        if cleaned_content:
            doc.page_content = cleaned_content
            cleaned_documents.append(doc)

    return cleaned_documents


def semantic_chunk_documents(documents):

    log_print(f"Total documents before cleaning: {len(documents)}")

    # Step 1: Clean documents
    cleaned_documents = clean_documents(documents)

    log_print(f"Total documents after cleaning: {len(cleaned_documents)}")

    # Step 2: Token-aware chunking
    splitter = TokenTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        encoding_name="gpt2",
    )

    chunks = splitter.split_documents(cleaned_documents)

    log_print(f"Total chunks created: {len(chunks)}")

    return chunks