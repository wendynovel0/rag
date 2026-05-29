from groq import Groq
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import Callable
import os

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "No se encontró la variable GROQ_API_KEY en el archivo .env. "
        "Crea el archivo .env con tu clave antes de ejecutar el sistema."
    )

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT: str = """Eres FitIA, una asistente experta en nutrición, fitness y bienestar que ayuda a los usuarios a alcanzar sus objetivos de salud de forma segura y efectiva.

REGLAS:
- Responde ÚNICAMENTE con información del contexto proporcionado.
- NUNCA empieces tu respuesta con saludos como "Buenos días", "Hola", "Buenas tardes" o similares. Comienza directamente con "Claro,", "Aquí tienes," o con la respuesta.
- Si la información no está en el contexto, di exactamente: "Hmm, no tengo información sobre eso en mi base de conocimiento aún. ¿Tienes alguna otra duda sobre nutrición, ejercicio o bienestar?"
- NUNCA inventes calorías, macronutrientes, pesos, repeticiones ni cualquier dato numérico que no esté en el contexto.
- Usa un tono motivador, cercano y positivo, como un entrenador o nutriólogo de confianza.
- Estructura tus respuestas con pasos claros, listas o tablas cuando sea útil (por ejemplo, para rutinas o planes de comida).
- Da respuestas completas con contexto: no solo des el dato, explica el porqué cuando puedas.
- Si el usuario menciona una condición médica (diabetes, hipertensión, embarazo, etc.), recuérdale amablemente que consulte a un profesional de salud antes de hacer cambios en su dieta o rutina.
- PROHIBIDO inventar información que no esté en el contexto."""


def create_rag(db: Chroma) -> Callable[[str], str]:
    """
    Crea y devuelve la función principal del sistema RAG con historial de conversación.

    Construye un closure que mantiene el historial del chat en memoria durante la sesión.
    Cada vez que el usuario hace una pregunta, se recuperan los 5 chunks más relevantes
    de ChromaDB, se inyectan como contexto al LLM y se genera una respuesta fundamentada
    únicamente en esa información.

    Args:
        db (Chroma): Base de datos vectorial con los embeddings de los documentos de
            nutrición, fitness y bienestar.

    Returns:
        Callable[[str], str]: Función `ask(question)` que acepta una pregunta en texto
            y devuelve la respuesta generada por el LLM como string.
    """
    historial: list[dict[str, str]] = []

    def ask(question: str) -> str:
        """
        Procesa una pregunta del usuario y devuelve la respuesta de FitIA.

        Busca los fragmentos de documentos más relevantes para la pregunta mediante
        similitud semántica, construye el prompt con ese contexto e invoca al LLM.
        La pregunta y la respuesta se agregan al historial para mantener coherencia
        en conversaciones de múltiples turnos. Si ocurre un error en la búsqueda o
        en la llamada al LLM, se devuelve un mensaje de error amigable sin romper
        el loop de conversación ni contaminar el historial.

        Args:
            question (str): Pregunta en lenguaje natural escrita por el usuario.

        Returns:
            str: Respuesta generada por el LLM basada exclusivamente en el contexto
                recuperado de los documentos, o mensaje de error si algo falla.
        """
        if not question.strip():
            return "Parece que tu pregunta llegó vacía. ¿Puedes escribirla de nuevo?"

        # --- Búsqueda semántica en ChromaDB ---
        try:
            docs = db.similarity_search("query: " + question, k=5)
            context: str = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return (
                f"Tuve un problema buscando en mis documentos. "
                f"Intenta de nuevo en un momento. (Detalle técnico: {e})"
            )

        # --- Construcción del mensaje y llamada al LLM ---
        mensaje_usuario: dict[str, str] = {
            "role": "user",
            "content": f"Contexto:\n{context}\n\nPregunta: {question}\n\nRespuesta (solo basada en el contexto):"
        }
        historial.append(mensaje_usuario)

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + historial
            )
            respuesta: str = response.choices[0].message.content
        except Exception as e:
            # Revertir el mensaje del usuario para no contaminar el historial
            historial.pop()
            return (
                f"Hubo un error al contactar al modelo de lenguaje. "
                f"Verifica tu conexión o API key e intenta de nuevo. (Detalle: {e})"
            )

        historial.append({"role": "assistant", "content": respuesta})
        return respuesta

    return ask
