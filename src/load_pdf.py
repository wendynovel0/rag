from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(folder_path="data"):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())

    return documents