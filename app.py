from src.load_pdf import load_pdf
from src.chunking import split_text
from src.embeddings import get_embeddings
from src.vector_db import create_db, load_db
from src.rag_chain import create_rag
from src.logger import init_log, measure, print_summary
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


def initialize_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Inicializa la base de datos vectorial, creándola o cargándola según corresponda.

    Si la carpeta 'chroma_db' no existe, ejecuta el pipeline completo de ingesta:
    carga los PDFs de nutrición y fitness, los fragmenta en chunks y genera los
    embeddings. Si ya existe, carga la base de datos persistida en disco para
    ahorrar tiempo de procesamiento en ejecuciones posteriores.

    Args:
        embeddings (HuggingFaceEmbeddings): Modelo de embeddings a usar para vectorizar
            o para cargar la base existente con las dimensiones correctas.

    Returns:
        Chroma: Instancia de la base de datos vectorial lista para consultas.

    Raises:
        RuntimeError: Si alguna etapa del pipeline de ingesta falla.
    """
    if not os.path.exists("chroma_db"):
        print("📄 Cargando PDFs...")
        documents = load_pdf()

        print("✂️ Dividiendo texto...")
        chunks = split_text(documents)

        print("🧠 Creando base vectorial...")
        return create_db(chunks, embeddings)
    else:
        print("📦 Cargando base vectorial existente...")
        return load_db(embeddings)


def main() -> None:
    """
    Punto de entrada principal del asistente FitIA.

    Orquesta la inicialización del sistema RAG con documentos de nutrición y fitness,
    y ejecuta el loop de conversación en la terminal. Cada consulta se mide en
    milisegundos y se registra en 'latency_log.csv' para verificar el cumplimiento
    del requisito de p95 < 100ms. Al salir, se imprime un resumen de latencias.
    """
    # --- Inicialización del log de latencias ---
    init_log()

    # --- Inicialización del modelo de embeddings ---
    try:
        embeddings: HuggingFaceEmbeddings = get_embeddings()
    except RuntimeError as e:
        print(f"\n❌ Error cargando embeddings: {e}")
        return

    # --- Inicialización de la base de datos vectorial ---
    try:
        db: Chroma = initialize_db(embeddings)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Error inicializando la base de datos: {e}")
        return

    # --- Inicio del sistema RAG ---
    print("🤖 Iniciando FitIA...")
    qa = measure(create_rag(db))  # measure() envuelve ask() para medir latencia

    print("\n💪 FitIA: ¡Hola! Soy FitIA, tu asistente de nutrición y fitness. ¿En qué te puedo ayudar hoy?\n")

    # --- Loop de conversación ---
    while True:
        try:
            question: str = input("\nPregunta (o escribe 'salir'): ")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 ¡Mucho ánimo y a seguir con los objetivos!")
            break

        if question.lower() == "salir":
            print("👋 ¡Mucho ánimo y a seguir con los objetivos!")
            break

        response: str = qa(question)
        print("\n🧠 FitIA:\n")
        print(response)

    # --- Resumen de latencias al terminar ---
    print_summary()


if __name__ == "__main__":
    main()
