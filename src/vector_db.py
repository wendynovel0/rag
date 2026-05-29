from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import re


def clean_text(text: str) -> str:
    """
    Limpia un string eliminando caracteres Unicode problemáticos (surrogates).

    Los PDFs a veces contienen caracteres surrogate inválidos que rompen ChromaDB
    al intentar persistir los embeddings. Esta función los elimina de forma segura
    sin perder el resto del contenido.

    Args:
        text (str): Texto crudo extraído de un PDF.

    Returns:
        str: Texto limpio, sin surrogates ni espacios extra al inicio/fin.
    """
    try:
        text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')
        text = re.sub(r'[\ud800-\udfff]', '', text)
        return text.strip()
    except Exception:
        # Si la limpieza falla por algún encoding raro, devolver string vacío
        # para que el chunk sea descartado en el filtro posterior
        return ""


def create_db(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Crea y persiste una base de datos vectorial ChromaDB a partir de los chunks de texto.

    Filtra chunks vacíos, limpia los textos de caracteres problemáticos y genera los
    embeddings para cada fragmento. La base de datos se guarda en disco en la carpeta
    'chroma_db' para poder reutilizarla sin reprocesar los PDFs en ejecuciones futuras.

    Args:
        chunks (list[Document]): Lista de fragmentos de texto generados por split_text().
        embeddings (HuggingFaceEmbeddings): Modelo de embeddings cargado con get_embeddings().

    Returns:
        Chroma: Instancia de la base de datos vectorial lista para hacer búsquedas.

    Raises:
        ValueError: Si después de filtrar no quedan chunks válidos para indexar.
        RuntimeError: Si ChromaDB falla al crear o persistir la base de datos.
    """
    clean_chunks: list[Document] = [
        chunk for chunk in chunks
        if chunk.page_content and isinstance(chunk.page_content, str) and chunk.page_content.strip()
    ]

    if not clean_chunks:
        raise ValueError(
            "No hay chunks válidos para indexar en ChromaDB. "
            "Revisa que los PDFs tengan texto extraíble (no sean imágenes escaneadas)."
        )

    print(f"✅ Chunks válidos: {len(clean_chunks)}")

    texts: list[str] = [clean_text(doc.page_content) for doc in clean_chunks]
    metadatas: list[dict] = [doc.metadata for doc in clean_chunks]

    # Descartar textos que quedaron vacíos tras la limpieza
    valid_pairs = [(t, m) for t, m in zip(texts, metadatas) if t]
    if not valid_pairs:
        raise ValueError("Todos los chunks quedaron vacíos después de la limpieza de texto.")

    texts, metadatas = zip(*valid_pairs)

    try:
        db = Chroma.from_texts(
            texts=list(texts),
            embedding=embeddings,
            metadatas=list(metadatas),
            persist_directory="chroma_db"
        )
        return db
    except Exception as e:
        raise RuntimeError(
            f"Error al crear la base de datos vectorial en ChromaDB: {e}"
        ) from e


def load_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Carga una base de datos vectorial ChromaDB previamente creada desde disco.

    Evita tener que reprocesar y reembeber los PDFs en cada ejecución. Se usa
    cuando la carpeta 'chroma_db' ya existe en el sistema de archivos.

    Args:
        embeddings (HuggingFaceEmbeddings): Modelo de embeddings, debe ser el mismo
            que se usó al crear la base de datos para que las dimensiones coincidan.

    Returns:
        Chroma: Instancia de la base de datos vectorial lista para hacer búsquedas.

    Raises:
        RuntimeError: Si ChromaDB no puede leer los datos desde disco.
    """
    try:
        return Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar la base de datos desde 'chroma_db'. "
            f"Intenta borrar la carpeta y volver a ejecutar para regenerarla.\nDetalle: {e}"
        ) from e
