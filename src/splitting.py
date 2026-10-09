# Splitting

from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(documents):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=80)
    chunks=splitter.split_documents(documents)
    return chunks