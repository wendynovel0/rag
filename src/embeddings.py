from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Carga y devuelve el modelo de embeddings multilingüe para vectorizar texto.

    Usa el modelo 'intfloat/multilingual-e5-base' de HuggingFace, optimizado para
    búsqueda semántica en múltiples idiomas incluyendo español. Los embeddings se
    normalizan para que la similitud coseno funcione correctamente con ChromaDB.

    Returns:
        HuggingFaceEmbeddings: Instancia del modelo de embeddings lista para usar.

    Raises:
        RuntimeError: Si el modelo no se puede cargar (sin conexión, nombre incorrecto, etc.).
    """
    try:
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo de embeddings 'multilingual-e5-base'. "
            f"Verifica tu conexión a internet o que sentence-transformers esté instalado.\nDetalle: {e}"
        ) from e
