"""
Motor de Function Calling HÍBRIDO para FitIA — Sistema Unificado.

Combina Function Calling (datos del usuario) y RAG (conocimiento de nutrición/fitness).
El LLM decide automáticamente cuándo usar herramientas vs conocimiento de documentos.

Flujo por turno:
  1. El mensaje llega al LLM con las herramientas disponibles.
  2. Si el LLM llama una función → se ejecuta localmente y se devuelve el resultado.
  3. Si el LLM responde con texto → puede usar contexto RAG inyectado en el system prompt.
"""

import json
from groq import Groq
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import Callable, Optional
import os

from src.functions import (
    crear_usuario,
    obtener_perfil,
    buscar_usuario_por_nombre,
    registrar_comida,
    consultar_calorias_hoy,
    eliminar_ultima_comida,
    registrar_ejercicio,
    consultar_ejercicios_hoy,
    calcular_balance_calorico,
    guardar_plan_semanal,
    consultar_historial,
)

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("No se encontró GROQ_API_KEY en el archivo .env.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Mapa de funciones — el dispatcher las busca aquí por nombre
# ---------------------------------------------------------------------------
FUNCTIONS_MAP: dict[str, Callable] = {
    "crear_usuario": crear_usuario,
    "obtener_perfil": obtener_perfil,
    "buscar_usuario_por_nombre": buscar_usuario_por_nombre,
    "registrar_comida": registrar_comida,
    "consultar_calorias_hoy": consultar_calorias_hoy,
    "eliminar_ultima_comida": eliminar_ultima_comida,
    "registrar_ejercicio": registrar_ejercicio,
    "consultar_ejercicios_hoy": consultar_ejercicios_hoy,
    "calcular_balance_calorico": calcular_balance_calorico,
    "guardar_plan_semanal": guardar_plan_semanal,
    "consultar_historial": consultar_historial,
}

# ---------------------------------------------------------------------------
# Catálogo de herramientas en formato JSON Schema para el LLM
# ---------------------------------------------------------------------------
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "crear_usuario",
            "description": (
                "Registra un nuevo perfil de usuario en FitIA. "
                "Usar cuando el usuario quiera crear su perfil o registrarse. "
                "Pedir nombre, edad, peso, altura y objetivo si no los proporcionó. "
                "NUNCA inventar estos datos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre":    {"type": "string",  "description": "Nombre completo del usuario."},
                    "edad":      {"type": "integer", "description": "Edad en años. DEBE ser un número entero real proporcionado por el usuario. NO llamar si no se tiene este valor."},
                    "peso_kg":   {"type": "number",  "description": "Peso actual en kilogramos. DEBE ser un número real proporcionado por el usuario. NO llamar si no se tiene este valor."},
                    "altura_cm": {"type": "number",  "description": "Altura en centímetros. DEBE ser un número real proporcionado por el usuario. NO llamar si no se tiene este valor."},
                    "objetivo":  {"type": "string",  "description": "Meta: 'bajar peso', 'subir peso', 'mantener peso', 'ganar músculo' o 'mejorar resistencia'."}
                },
                "required": ["nombre", "edad", "peso_kg", "altura_cm", "objetivo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_perfil",
            "description": "Consulta el perfil completo de un usuario por su ID numérico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID numérico del usuario."}
                },
                "required": ["usuario_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_usuario_por_nombre",
            "description": (
                "Busca un usuario registrado por su nombre y devuelve su ID. "
                "USAR SIEMPRE cuando el usuario mencione su nombre pero no su ID. "
                "Esto permite identificarlo antes de registrar comidas, ejercicios o consultar datos. "
                "Si no recuerda su ID, busca por nombre primero. "
                "IMPORTANTE: Si la función devuelve más de un usuario, "
                "SIEMPRE pregunta al usuario cuál es el suyo antes de continuar con cualquier otra acción."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string", "description": "Nombre o parte del nombre a buscar."}
                },
                "required": ["nombre"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "registrar_comida",
            "description": (
                "Registra un alimento consumido en el log del día. Usar cuando el usuario diga que comió algo. "
                "Si el usuario menciona varios alimentos en un mismo mensaje, llama esta función una vez por cada alimento. "
                "Los macros (proteínas, carbohidratos, grasas) son opcionales — si el usuario no los dio, usar 0. "
                "NUNCA pedir macros al usuario si no los proporcionó."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id":      {"type": "integer", "description": "ID del usuario."},
                    "alimento":        {"type": "string",  "description": "Nombre del alimento."},
                    "cantidad_g":      {"type": "number",  "description": "Cantidad en gramos."},
                    "calorias":        {"type": "number",  "description": "Calorías totales de esa porción."},
                    "proteinas_g":     {"type": "number", "description": "Gramos de proteína. OPCIONAL — usar 0 si el usuario no lo mencionó. NUNCA preguntar al usuario por este dato."},
                    "carbohidratos_g": {"type": "number", "description": "Gramos de carbohidratos. OPCIONAL — usar 0 si el usuario no lo mencionó. NUNCA preguntar al usuario por este dato."},
                    "grasas_g":        {"type": "number", "description": "Gramos de grasa. OPCIONAL — usar 0 si el usuario no lo mencionó. NUNCA preguntar al usuario por este dato."}
                },
                "required": ["usuario_id", "alimento", "cantidad_g", "calorias"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_calorias_hoy",
            "description": "Muestra el resumen de calorías y macronutrientes consumidos hoy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."}
                },
                "required": ["usuario_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "eliminar_ultima_comida",
            "description": "Elimina el último alimento registrado hoy. Usar cuando el usuario diga que se equivocó.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."}
                },
                "required": ["usuario_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "registrar_ejercicio",
            "description": "Registra un ejercicio realizado hoy por el usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id":        {"type": "integer", "description": "ID del usuario."},
                    "ejercicio":         {"type": "string",  "description": "Nombre del ejercicio."},
                    "series":            {"type": "integer", "description": "Número de series. Usar 0 para cardio."},
                    "repeticiones":      {"type": "integer", "description": "Repeticiones por serie. Usar 0 para cardio."},
                    "duracion_min":      {"type": "number",  "description": "Duración en minutos."},
                    "calorias_quemadas": {"type": "number",  "description": "Calorías estimadas quemadas."}
                },
                "required": ["usuario_id", "ejercicio"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_ejercicios_hoy",
            "description": "Muestra todos los ejercicios registrados hoy y el total de calorías quemadas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."}
                },
                "required": ["usuario_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_balance_calorico",
            "description": "Calcula el balance calórico del día (ingesta menos gasto) y estima la TMB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."}
                },
                "required": ["usuario_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "guardar_plan_semanal",
            "description": "Guarda un plan semanal alimentario o de entrenamiento para el usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."},
                    "tipo":       {"type": "string",  "description": "Tipo: 'alimentario' o 'entrenamiento'."},
                    "contenido":  {"type": "string",  "description": "Texto completo del plan semanal."}
                },
                "required": ["usuario_id", "tipo", "contenido"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_historial",
            "description": "Muestra el resumen de calorías y ejercicio de los últimos N días.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."},
                    "dias":       {"type": "integer", "description": "Número de días a consultar (1-30). Por defecto 7."}
                },
                "required": ["usuario_id"]
            }
        }
    }
]


def _build_system_prompt(rag_context: Optional[str] = None) -> str:
    """
    Construye el system prompt dinámico del agente híbrido.

    Args:
        rag_context: Texto de contexto recuperado de ChromaDB, o None.

    Returns:
        str: System prompt completo.
    """
    base = """Eres FitIA, una asistente experta en nutrición, fitness y bienestar con acceso a herramientas para gestionar datos del usuario.

REGLAS CRÍTICAS:
1. IDENTIFICACIÓN OBLIGATORIA: Antes de registrar comidas, ejercicios, consultar datos o calcular balances,
   DEBES saber el ID del usuario. Si el usuario dice su nombre pero no su ID, usa buscar_usuario_por_nombre
   primero. Si no está registrado, usa crear_usuario. NUNCA asumas un ID.

2. DATOS FALTANTES: NUNCA llames una función si algún parámetro requerido no fue proporcionado
   explícitamente por el usuario. Si faltan datos, PREGUNTA primero y espera la respuesta.
   PROHIBIDO pasar strings como "desconocido", "valor desconocido" o similares — eso causará un error.
   Solo llama la función cuando tengas TODOS los valores reales del usuario.

3. HERRAMIENTAS: Usa las herramientas cuando el usuario quiera:
   - Crear o consultar su perfil (crear_usuario, obtener_perfil, buscar_usuario_por_nombre)
   - Registrar comidas o ejercicios (registrar_comida, registrar_ejercicio)
   - Ver su progreso (consultar_calorias_hoy, consultar_ejercicios_hoy, calcular_balance_calorico)
   - Ver historial o guardar planes (consultar_historial, guardar_plan_semanal)

4. PREGUNTAS DE CONOCIMIENTO: Para preguntas sobre nutrición, dietas, ejercicio o bienestar,
   responde ÚNICAMENTE con la información del CONTEXTO DE DOCUMENTOS que se te proporciona abajo.
   Si la información no está en ese contexto, responde exactamente:
   "Hmm, no tengo información sobre eso en mi base de conocimiento. ¿Tienes otra duda sobre 
   nutrición, ejercicio o bienestar?"
   NUNCA respondas preguntas fuera del dominio fitness/nutrición (historia, deportes, política, etc.)."""

    if rag_context:
        base += f"""

⚠️ CONTEXTO OBLIGATORIO — USA SOLO ESTO PARA RESPONDER PREGUNTAS DE CONOCIMIENTO:
{rag_context}

Si la respuesta no está en este contexto, admite que no tienes esa información. NUNCA uses conocimiento externo."""
    else:
        base += "\n\nResponde con tu conocimiento experto cuando no se necesiten herramientas."

    base += "\n\nSé conciso, motivador y claro. Tono amigable como entrenador de confianza."
    return base


def dispatch_function(name: str, arguments: dict) -> str:
    """
    Ejecuta la función local correspondiente al nombre recibido del LLM.

    Args:
        name      (str):  Nombre de la función a ejecutar.
        arguments (dict): Parámetros tal como los generó el LLM.

    Returns:
        str: Resultado serializado como JSON string.
    """
    if name not in FUNCTIONS_MAP:
        return json.dumps({"error": f"Función '{name}' no encontrada en el catálogo."})

    try:
        resultado = FUNCTIONS_MAP[name](**arguments)
        return json.dumps(resultado, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"error": f"Parámetros incorrectos para '{name}': {e}"})
    except Exception as e:
        return json.dumps({"error": f"Error inesperado al ejecutar '{name}': {e}"})


def create_hybrid_agent(db: Optional[Chroma] = None) -> Callable[[str], str]:
    """
    Crea el agente híbrido FitIA que combina Function Calling y RAG.

    Args:
        db (Optional[Chroma]): Base de datos vectorial con documentos de fitness/nutrición.

    Returns:
        Callable[[str], str]: Función chat(message) que procesa mensajes del usuario.
    """
    historial: list[dict] = []

    def chat(message: str) -> str:
        """
        Procesa un mensaje del usuario a través del agente híbrido.

        Flujo:
          1. Si hay db RAG, busca contexto relevante en los documentos.
          2. Construye el system prompt dinámico.
          3. Llama al LLM con herramientas disponibles.
          4. Si hay tool_calls → ejecuta funciones → segunda llamada al LLM.
          5. Devuelve respuesta final en lenguaje natural.

        Args:
            message (str): Mensaje del usuario.

        Returns:
            str: Respuesta de FitIA.
        """
        if not message.strip():
            return "Parece que tu mensaje llegó vacío. ¿Puedes escribirlo de nuevo?"

        # --- Paso 1: Recuperar contexto RAG si hay documentos ---
        rag_context: Optional[str] = None
        if db is not None:
            try:
                docs = db.similarity_search("query: " + message, k=4)
                if docs:
                    rag_context = "\n\n".join([doc.page_content for doc in docs])
            except Exception:
                rag_context = None  # RAG falla en silencio, no bloquear al usuario

        # --- Paso 2: Construir system prompt dinámico ---
        system_prompt = _build_system_prompt(rag_context)

        # --- Paso 3: Agregar mensaje al historial y llamar al LLM ---
        historial.append({"role": "user", "content": message})

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}] + historial,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )
        except Exception as e:
            error_str = str(e)
            # Si Groq rechazó el tool_call por JSON malformado, reintentar sin tools
            if "tool_use_failed" in error_str or "400" in error_str:
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": system_prompt}] + historial,
                        max_tokens=1024,
                    )
                except Exception as e2:
                    historial.pop()
                    return f"Error al contactar el modelo. (Detalle: {e2})"
            else:
                historial.pop()
                return f"Error al contactar el modelo. Verifica tu API key. (Detalle: {e})"

        msg = response.choices[0].message

        # --- Paso 4a: El LLM responde con texto directo (sin función) ---
        if not msg.tool_calls:
            respuesta: str = msg.content or "No tengo una respuesta para eso ahora mismo."
            historial.append({"role": "assistant", "content": respuesta})
            return respuesta

        # --- Paso 4b: El LLM decidió llamar una o más funciones ---
        historial.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        })

        # Ejecutar cada función y agregar resultados al historial
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args_raw = tool_call.function.arguments

            print(f"\n🔧 [Function Call] {fn_name}")
            print(f"   Args: {fn_args_raw}")

            # BUG CORREGIDO: manejar JSON inválido sin crashear
            try:
                fn_args = json.loads(fn_args_raw) if fn_args_raw else {}
            except json.JSONDecodeError as e:
                fn_args = {}
                print(f"   ⚠️  JSON inválido del LLM: {e}. Se usarán args vacíos.")

            resultado_str = dispatch_function(fn_name, fn_args)
            print(f"   Resultado: {resultado_str}")

            historial.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": resultado_str
            })

        # --- Paso 5: Segunda llamada al LLM con resultados de funciones ---
        try:
            response2 = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}] + historial,
                tools=TOOLS,
                tool_choice="none",  # No queremos más tool_calls en esta segunda vuelta
                max_tokens=1024,
            )
            respuesta_final: str = response2.choices[0].message.content or "Listo, acción completada."
        except Exception as e:
            # BUG CORREGIDO: devolver mensaje útil sin perder el contexto ya ejecutado
            respuesta_final = "Las funciones se ejecutaron correctamente, pero hubo un error generando la respuesta. Intenta preguntar de nuevo."
            print(f"   ⚠️  Error en segunda llamada al LLM: {e}")

        historial.append({"role": "assistant", "content": respuesta_final})
        return respuesta_final

    return chat


# ---------------------------------------------------------------------------
# Compatibilidad con modo sin RAG
# ---------------------------------------------------------------------------
def create_function_calling_agent() -> Callable[[str], str]:
    """Crea el agente en modo function-calling puro (sin RAG)."""
    return create_hybrid_agent(db=None)