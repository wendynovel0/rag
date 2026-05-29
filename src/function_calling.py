"""
Motor de Function Calling para FitIA — Fase 2.

Este módulo es el corazón de la Fase 2. Se encarga de:
  1. Definir el catálogo de herramientas en el formato que entiende el LLM.
  2. Enviar la pregunta del usuario al LLM con las herramientas disponibles.
  3. Interceptar el JSON que genera el LLM cuando decide llamar una función.
  4. Ejecutar la función local correspondiente con los parámetros recibidos.
  5. Devolver el resultado al LLM para que genere la respuesta final en lenguaje natural.
  6. Manejar errores en cada etapa sin contaminar el historial de conversación.
"""

import json
from groq import Groq
from dotenv import load_dotenv
from typing import Callable
import os

from src.functions import (
    crear_usuario,
    obtener_perfil,
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
# Mapa de funciones disponibles — el dispatcher las busca aquí por nombre
# ---------------------------------------------------------------------------
FUNCTIONS_MAP: dict[str, Callable] = {
    "crear_usuario": crear_usuario,
    "obtener_perfil": obtener_perfil,
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
            "description": "Registra un nuevo usuario con su perfil de salud y objetivo fitness. Usar cuando el usuario quiere empezar a usar FitIA o crear su perfil.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre":     {"type": "string",  "description": "Nombre completo del usuario."},
                    "edad":       {"type": "integer", "description": "Edad en años."},
                    "peso_kg":    {"type": "number",  "description": "Peso actual en kilogramos."},
                    "altura_cm":  {"type": "number",  "description": "Altura en centímetros."},
                    "objetivo":   {"type": "string",  "description": "Meta de salud: 'bajar peso', 'subir peso', 'mantener peso', 'ganar músculo' o 'mejorar resistencia'."}
                },
                "required": ["nombre", "edad", "peso_kg", "altura_cm", "objetivo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_perfil",
            "description": "Consulta el perfil completo de un usuario: nombre, edad, peso, altura y objetivo.",
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
            "name": "registrar_comida",
            "description": "Registra un alimento consumido por el usuario en el log del día actual. Usar cuando el usuario diga que comió algo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id":      {"type": "integer", "description": "ID del usuario."},
                    "alimento":        {"type": "string",  "description": "Nombre del alimento consumido."},
                    "cantidad_g":      {"type": "number",  "description": "Cantidad consumida en gramos."},
                    "calorias":        {"type": "number",  "description": "Calorías totales de esa porción."},
                    "proteinas_g":     {"type": "number",  "description": "Gramos de proteína. Usar 0 si no se sabe."},
                    "carbohidratos_g": {"type": "number",  "description": "Gramos de carbohidratos. Usar 0 si no se sabe."},
                    "grasas_g":        {"type": "number",  "description": "Gramos de grasa. Usar 0 si no se sabe."}
                },
                "required": ["usuario_id", "alimento", "cantidad_g", "calorias"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_calorias_hoy",
            "description": "Muestra el resumen de calorías y macronutrientes consumidos hoy por el usuario.",
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
            "description": "Elimina el último alimento registrado hoy por el usuario. Usar cuando el usuario diga que se equivocó al registrar.",
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
            "description": "Registra un ejercicio realizado por el usuario hoy. Usar cuando el usuario diga que hizo ejercicio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id":        {"type": "integer", "description": "ID del usuario."},
                    "ejercicio":         {"type": "string",  "description": "Nombre del ejercicio realizado."},
                    "series":            {"type": "integer", "description": "Número de series. Usar 0 para cardio."},
                    "repeticiones":      {"type": "integer", "description": "Repeticiones por serie. Usar 0 para cardio."},
                    "duracion_min":      {"type": "number",  "description": "Duración en minutos. Usar 0 para ejercicios de fuerza."},
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
            "description": "Muestra todos los ejercicios que el usuario registró hoy y el total de calorías quemadas.",
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
            "description": "Calcula el balance calórico del día: calorías consumidas menos calorías quemadas. También estima la tasa metabólica basal (TMB).",
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
                    "tipo":       {"type": "string",  "description": "Tipo de plan: 'alimentario' o 'entrenamiento'."},
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
            "description": "Muestra el resumen de calorías y ejercicio de los últimos N días del usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "usuario_id": {"type": "integer", "description": "ID del usuario."},
                    "dias":       {"type": "integer", "description": "Número de días hacia atrás a consultar (1-30). Por defecto 7."}
                },
                "required": ["usuario_id"]
            }
        }
    }
]

SYSTEM_PROMPT_FC: str = """Eres FitIA, una asistente experta en nutrición y fitness con acceso a herramientas para registrar y consultar datos del usuario.

REGLAS DE USO DE HERRAMIENTAS:
- Usa las herramientas siempre que el usuario quiera registrar o consultar datos (comidas, ejercicios, perfil, balance, historial).
- Si el usuario no te ha dado su ID de usuario y lo necesitas, pregúntaselo antes de llamar cualquier herramienta. NO inventes un ID.
- Si falta cualquier dato requerido (nombre, peso, alimento, etc.), pregunta al usuario antes de ejecutar la función. NUNCA inventes datos.
- Cuando una función devuelva un error, explícaselo al usuario de forma amigable y sugiere cómo corregirlo.
- Para preguntas de conocimiento general sobre nutrición o fitness, responde con tu conocimiento sin usar herramientas.
- Sé conciso, motivador y claro en tus respuestas."""


def dispatch_function(name: str, arguments: dict) -> str:
    """
    Ejecuta la función local correspondiente al nombre recibido del LLM.

    Busca la función en el mapa de funciones disponibles y la llama con los
    argumentos proporcionados. Captura cualquier excepción para que un error
    en una función no rompa el flujo de conversación.

    Args:
        name      (str):  Nombre de la función a ejecutar (debe existir en FUNCTIONS_MAP).
        arguments (dict): Diccionario de parámetros tal como los generó el LLM.

    Returns:
        str: Resultado de la función serializado como JSON string, listo para
             enviarse al LLM como tool_result. En caso de error, devuelve un
             JSON con la clave "error".
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


def create_function_calling_agent() -> Callable[[str], str]:
    """
    Crea y devuelve el agente de Function Calling con historial de conversación.

    Construye un closure que mantiene el historial del chat en memoria.
    En cada turno: envía el mensaje al LLM con las herramientas disponibles,
    intercepta si el LLM decide llamar una función, la ejecuta localmente,
    y devuelve el resultado al LLM para la respuesta final.

    Returns:
        Callable[[str], str]: Función `chat(message)` que procesa mensajes del
            usuario y devuelve respuestas en lenguaje natural.
    """
    historial: list[dict] = []

    def chat(message: str) -> str:
        """
        Procesa un mensaje del usuario a través del agente de Function Calling.

        Flujo interno:
          1. Agrega el mensaje al historial.
          2. Llama al LLM con las herramientas disponibles.
          3. Si el LLM responde con texto → lo devuelve directamente.
          4. Si el LLM genera una tool_call → intercepta el JSON, ejecuta
             la función local, agrega el resultado al historial y vuelve
             a llamar al LLM para obtener la respuesta final en lenguaje natural.

        Args:
            message (str): Mensaje en lenguaje natural del usuario.

        Returns:
            str: Respuesta final del LLM en lenguaje natural, o mensaje de
                 error amigable si algo falla en el proceso.
        """
        if not message.strip():
            return "Parece que tu mensaje llegó vacío. ¿Puedes escribirlo de nuevo?"

        historial.append({"role": "user", "content": message})

        # --- Primera llamada al LLM ---
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT_FC}] + historial,
                tools=TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            historial.pop()  # Revertir para no contaminar el historial
            return f"Error al contactar el modelo. Verifica tu API key. (Detalle: {e})"

        msg = response.choices[0].message

        # --- El LLM responde con texto directo (sin llamar función) ---
        if not msg.tool_calls:
            respuesta: str = msg.content or "No tengo una respuesta para eso ahora mismo."
            historial.append({"role": "assistant", "content": respuesta})
            return respuesta

        # --- El LLM decidió llamar una o más funciones ---
        # Agregar el mensaje del asistente con las tool_calls al historial
        historial.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            }
            for tc in msg.tool_calls
        ]})

        # Ejecutar cada función y agregar resultados al historial
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args_raw = tool_call.function.arguments

            print(f"\n🔧 [Function Call interceptado]")
            print(f"   Función  : {fn_name}")
            print(f"   Argumentos: {fn_args_raw}")

            try:
                fn_args = json.loads(fn_args_raw)
            except json.JSONDecodeError as e:
                fn_args = {}
                print(f"   ⚠️  Error al parsear argumentos: {e}")

            resultado_str = dispatch_function(fn_name, fn_args)
            print(f"   Resultado : {resultado_str}")

            historial.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": resultado_str
            })

        # --- Segunda llamada al LLM con el resultado de las funciones ---
        try:
            response2 = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT_FC}] + historial,
                tools=TOOLS,
                tool_choice="auto"
            )
            respuesta_final: str = response2.choices[0].message.content or "Listo, acción completada."
        except Exception as e:
            return f"Las funciones se ejecutaron, pero hubo un error al generar la respuesta final. (Detalle: {e})"

        historial.append({"role": "assistant", "content": respuesta_final})
        return respuesta_final

    return chat
