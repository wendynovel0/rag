from src.load_pdf import load_pdf
from src.chunking import split_text
from src.embeddings import get_embeddings
from src.vector_db import create_db, load_db
from src.rag_chain import create_rag

import os

def main():

    embeddings = get_embeddings()

    if not os.path.exists("chroma_db"):

        print("📄 Cargando PDFs...")
        documents = load_pdf()

        print("✂️ Dividiendo texto...")
        chunks = split_text(documents)

        print("🧠 Creando base vectorial...")
        db = create_db(chunks, embeddings)

    else:
        print("📦 Cargando base vectorial...")
        db = load_db(embeddings)

    print("🤖 Iniciando RAG...")
    qa = create_rag(db)

    print("\n🌺 Maya: ¡Hola! Soy Maya, tu asistente de vida independiente en Yucatán. ¿En qué te puedo ayudar hoy?\n")

    while True:

        question = input("\nPregunta (o salir): ")

        if question.lower() == "salir":
            break

        response = qa(question)

        print("\n🧠 Respuesta:\n")
        print(response)

if __name__ == "__main__":
    main()