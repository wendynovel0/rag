"""
Punto de entrada de la Fase 2 — FitIA Function Calling.

Ejecuta el agente de function calling en modo terminal. El usuario puede
interactuar con FitIA para registrar comidas, ejercicios, consultar su
balance calórico e historial, todo mediante lenguaje natural.

Uso:
    python main_fc.py
"""

from src.database import init_db
from src.function_calling import create_function_calling_agent


def main() -> None:
    """
    Inicializa la base de datos y arranca el loop de conversación del agente.

    Crea las tablas SQLite si no existen y lanza el agente de Function Calling.
    Muestra en consola el flujo interno cada vez que el LLM invoca una función:
    nombre, argumentos JSON y resultado — útil para el video de seguimiento de procesos.
    """
    print("🗄️  Inicializando base de datos...")
    try:
        init_db()
        print("✅ Base de datos lista.")
    except RuntimeError as e:
        print(f"❌ Error al inicializar la base de datos: {e}")
        return

    print("🤖 Iniciando agente FitIA (Function Calling)...")
    agent = create_function_calling_agent()

    print("\n💪 FitIA: ¡Hola! Soy FitIA. Puedo registrar tus comidas y ejercicios,")
    print("   calcular tu balance calórico y mucho más. ¿Con qué empezamos?\n")
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

        respuesta: str = agent(user_input)
        print(f"\nFitIA: {respuesta}")
        print("─" * 60)


if __name__ == "__main__":
    main()
