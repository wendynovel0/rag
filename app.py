"""
Punto de entrada principal de FitIA — Sistema Híbrido (RAG + Function Calling).

Este archivo inicializa el sistema completo:
  - Base de datos SQLite para datos del usuario (Function Calling)
  - Base de datos vectorial ChromaDB con documentos de fitness/nutrición (RAG)
  - Agente híbrido que combina ambos automáticamente según el tipo de pregunta

El usuario puede en la misma conversación:
  - Crear su perfil y registrar comidas/ejercicios (Function Calling)
  - Preguntar sobre nutrición, rutinas y salud (RAG con documentos especializados)

Latencias registradas en 'latency_log.csv' para verificar cumplimiento de p95 < 1 seg.
"""

import os
from src.load_pdf import load_pdf
from src.chunking import split_text
from src.embeddings import get_embeddings
from src.vector_db import create_db, load_db
from src.database import init_db
from src.function_calling import create_hybrid_agent
from src.logger import init_log, measure, print_summary
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_vector_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Inicializa ChromaDB creándola desde PDFs o cargándola si ya existe.

    Args:
        embeddings: Modelo de embeddings a usar.

    Returns:
        Chroma: Base de datos vectorial lista para consultas.
    """
    if not os.path.exists("chroma_db"):
        print("📄 Cargando PDFs...")
        documents = load_pdf()

        print("✂️  Dividiendo texto en chunks...")
        chunks = split_text(documents)

        print("🧠 Creando base vectorial ChromaDB...")
        return create_db(chunks, embeddings)
    else:
        print("📦 Cargando base vectorial existente desde disco...")
        return load_db(embeddings)


def main() -> None:
    """
    Orquesta el arranque del sistema FitIA híbrido y ejecuta el loop de conversación.

    Inicializa en orden:
      1. Log de latencias
      2. Base de datos SQLite (usuarios, comidas, ejercicios, planes)
      3. Modelo de embeddings
      4. Base de datos vectorial ChromaDB
      5. Agente híbrido (RAG + Function Calling)
    """
    # 1. Log de latencias
    init_log()

    # 2. Base de datos SQLite para function calling
    print("🗄️  Inicializando base de datos SQLite...")
    try:
        init_db()
        print("✅ Base de datos SQLite lista.")
    except RuntimeError as e:
        print(f"\n❌ Error inicializando SQLite: {e}")
        return

    # 3. Modelo de embeddings
    print("🔤 Cargando modelo de embeddings...")
    try:
        embeddings: HuggingFaceEmbeddings = get_embeddings()
        print("✅ Embeddings listos.")
    except RuntimeError as e:
        print(f"\n❌ Error cargando embeddings: {e}")
        return

    # 4. Base de datos vectorial
    try:
        db: Chroma = initialize_vector_db(embeddings)
        print("✅ ChromaDB lista.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Error inicializando ChromaDB: {e}")
        return

    # 5. Agente híbrido
    print("\n🤖 Iniciando agente FitIA híbrido (RAG + Function Calling)...")
    agent_fn = create_hybrid_agent(db=db)
    # Envolver con measure() para medir latencia
    agent = measure(agent_fn)

    print("\n💪 FitIA: ¡Hola! Soy FitIA, tu asistente de nutrición y fitness.")
    print("   Puedo responder preguntas sobre nutrición, registrar tus comidas y ejercicios,")
    print("   calcular tu balance calórico y mucho más.")
    print("   (Escribe 'salir' para terminar)\n")
    print("─" * 60)

    while True:
        try:
            user_input: str = input("\nTú: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 ¡Hasta luego! Sigue con tus objetivos 💪")
            break

        if user_input.lower() == "salir":
            print("👋 ¡Hasta luego! Sigue con tus objetivos 💪")
            break

        response: str = agent(user_input)
        print(f"\nFitIA: {response}")
        print("─" * 60)

    print_summary()


if __name__ == "__main__":
    main()