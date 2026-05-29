from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os


def load_pdf(folder_path: str = "data") -> list[Document]:
    """
    Carga todos los archivos PDF de una carpeta y los devuelve como documentos LangChain.

    Itera sobre los archivos en la carpeta indicada, filtra únicamente los que tienen
    extensión .pdf y los carga página por página usando PyPDFLoader. Cada página se
    convierte en un objeto Document con su contenido y metadatos (fuente, número de página).
    Los PDFs que fallen individualmente se omiten con una advertencia, sin detener el proceso.

    Args:
        folder_path (str): Ruta a la carpeta que contiene los PDFs. Por defecto "data".

    Returns:
        list[Document]: Lista de documentos LangChain, uno por página de cada PDF cargado.

    Raises:
        FileNotFoundError: Si la carpeta indicada no existe.
        ValueError: Si la carpeta no contiene ningún archivo PDF.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"La carpeta '{folder_path}' no existe. Crea la carpeta y coloca los PDFs dentro."
        )

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(
            f"No se encontraron archivos PDF en '{folder_path}'. "
            f"Asegúrate de que la carpeta contenga al menos un PDF."
        )

    documents: list[Document] = []

    for file in pdf_files:
        file_path = os.path.join(folder_path, file)
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            # Un PDF corrupto no debe detener toda la ingesta
            print(f"⚠️  No se pudo cargar '{file}', se omitirá. Detalle: {e}")

    if not documents:
        raise ValueError(
            "Ningún PDF pudo ser leído correctamente. Verifica que los archivos no estén corruptos."
        )

    return documents
