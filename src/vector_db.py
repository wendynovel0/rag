from langchain_chroma import Chroma
import re

def clean_text(text):
    # Elimina surrogates y caracteres problemáticos
    text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text.strip()

def create_db(chunks, embeddings):
    clean_chunks = []
    for chunk in chunks:
        if chunk.page_content and isinstance(chunk.page_content, str):
            text = chunk.page_content.strip()
            if text != "":
                clean_chunks.append(chunk)

    print(f"✅ Chunks válidos: {len(clean_chunks)}")

    texts = [clean_text(doc.page_content) for doc in clean_chunks]
    metadatas = [doc.metadata for doc in clean_chunks]

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="chroma_db"
    )
    return db


def load_db(embeddings):
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
