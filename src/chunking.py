from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_text(documents: list[Document]) -> list[Document]:
    """
    Divide una lista de documentos en chunks más pequeños con superposición semántica.

    Usa RecursiveCharacterTextSplitter para fragmentar el texto de forma inteligente,
    respetando límites naturales como párrafos, oraciones y palabras antes de cortar.
    El overlap evita que se pierda contexto entre un chunk y el siguiente.

    Args:
        documents (list[Document]): Lista de documentos LangChain cargados desde PDF u otra fuente.

    Returns:
        list[Document]: Lista de chunks limpios (sin contenido vacío), listos para embeber.

    Raises:
        ValueError: Si la lista de documentos está vacía.
        RuntimeError: Si el splitter falla al procesar los documentos.
    """
    if not documents:
        raise ValueError("La lista de documentos está vacía. Verifica que la carpeta 'data' contenga PDFs.")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks: list[Document] = splitter.split_documents(documents)
    except Exception as e:
        raise RuntimeError(f"Error al dividir los documentos en chunks: {e}") from e

    # Limpieza extra: descartar chunks que quedaron vacíos tras el split
    chunks_limpios = [c for c in chunks if c.page_content.strip() != ""]

    if not chunks_limpios:
        raise ValueError("El chunking no produjo ningún fragmento válido. Revisa el contenido de los PDFs.")

    return chunks_limpios
