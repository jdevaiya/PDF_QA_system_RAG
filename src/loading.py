from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

def loader(file):
    temp_dir=tempfile.gettempdir()
    temp_dir_path=os.path.join(temp_dir, file.name)
    
    with open(temp_dir_path, "wb")as f:
        f.write(file.getbuffer())
        
    if not os.path.exists(temp_dir_path):
        raise FileNotFoundError(f"File not found: {temp_dir_path}")
    
    loader=PyPDFLoader(temp_dir_path)
    documents=loader.load()
    return documents