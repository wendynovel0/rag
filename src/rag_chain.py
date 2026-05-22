from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def create_rag(db):
    historial = []

    def ask(question):
        docs = db.similarity_search("query: " + question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        historial.append({"role": "user", "content": f"""Contexto:
{context}

Pregunta: {question}

Respuesta (solo basada en el contexto):"""})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Eres Maya, una asistente amable y cercana que ayuda a jóvenes adultos en Yucatán a navegar trámites, finanzas y vida independiente.

REGLAS:
- Responde ÚNICAMENTE con información del contexto proporcionado.
- NUNCA empieces tu respuesta con saludos como "Buenos días", "Hola", "Buenos tardes" o similares. Hazlo con "Claro", "Aquí tienes".
- Si no está en el contexto, di: "Hmm, no tengo información sobre eso en mis documentos aún. ¿Tienes alguna otra pregunta sobre trámites, finanzas o vida independiente en Yucatán?"
- Usa un tono amigable, cercano y alentador, como si fuera una amiga que sabe mucho.
- Estructura tus respuestas con pasos claros o puntos cuando sea útil.
- Da respuestas completas, no solo una oración. Explica el contexto cuando puedas.
- PROHIBIDO inventar información que no esté en el contexto."""
                }
            ] + historial
        )

        respuesta = response.choices[0].message.content
        historial.append({"role": "assistant", "content": respuesta})

        return respuesta

    return ask