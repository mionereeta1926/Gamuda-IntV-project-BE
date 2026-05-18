from langchain_text_splitters import TokenTextSplitter


def semantic_chunk_documents(documents):
    # Token-aware splitting is better than raw character splitting because it
    # aligns chunks with the model's tokenization. This gives more consistent
    # length control and avoids cutting content in awkward positions.
    splitter = TokenTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        encoding_name="gpt2",
    )

    chunks = splitter.split_documents(documents)

    return chunks