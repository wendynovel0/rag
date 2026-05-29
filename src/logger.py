import time
import csv
import os
from datetime import datetime
from typing import Callable

LOG_FILE: str = "latency_log.csv"


def init_log() -> None:
    """
    Crea el archivo CSV de log si no existe, escribiendo la fila de encabezados.

    El archivo se crea en el directorio raíz del proyecto con el nombre 'latency_log.csv'.
    Si ya existe (de una ejecución anterior), no lo sobreescribe para preservar el historial.
    """
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "pregunta", "latencia_ms", "cumple_100ms"])
        except Exception as e:
            print(f"⚠️  No se pudo crear el archivo de log: {e}")


def log_latency(question: str, latency_ms: float) -> None:
    """
    Registra una medición de latencia en el archivo CSV de log.

    Guarda la pregunta, el tiempo de respuesta en milisegundos y un booleano
    que indica si cumplió el umbral de 100ms requerido por la rúbrica.

    Args:
        question (str): Pregunta que hizo el usuario.
        latency_ms (float): Tiempo total de respuesta en milisegundos.
    """
    cumple: bool = latency_ms < 100.0
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, question[:80], f"{latency_ms:.2f}", cumple])
    except Exception as e:
        print(f"⚠️  No se pudo guardar el log de latencia: {e}")


def measure(func: Callable[[str], str]) -> Callable[[str], str]:
    """
    Decorador / wrapper que mide el tiempo de ejecución de la función ask() del RAG.

    Envuelve la función recibida para medir cuántos milisegundos tarda en responder,
    imprime el resultado en consola con un indicador visual (✅ / ⚠️) y lo guarda en
    el CSV de log para análisis posterior (p95, promedio, etc.).

    Args:
        func (Callable[[str], str]): Función ask() generada por create_rag().

    Returns:
        Callable[[str], str]: Nueva función con la misma firma que mide y loggea
            automáticamente cada llamada.
    """
    def wrapper(question: str) -> str:
        start: float = time.perf_counter()
        result: str = func(question)
        end: float = time.perf_counter()

        latency_ms: float = (end - start) * 1000
        cumple: bool = latency_ms < 100.0

        icono: str = "✅" if cumple else "⚠️ "
        print(f"\n{icono} Latencia: {latency_ms:.2f}ms {'(< 100ms ✓)' if cumple else '(> 100ms — revisar)'}")

        log_latency(question, latency_ms)
        return result

    return wrapper


def print_summary() -> None:
    """
    Lee el CSV de log y muestra un resumen de latencias en consola al terminar la sesión.

    Calcula el promedio, el mínimo, el máximo y el percentil 95 (p95) de todas las
    consultas registradas. Indica si el sistema cumple con el requisito de p95 < 100ms.
    Si el archivo no existe o está vacío, muestra un mensaje informativo.
    """
    if not os.path.exists(LOG_FILE):
        print("\n📊 No hay datos de latencia registrados aún.")
        return

    try:
        latencias: list[float] = []
        with open(LOG_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    latencias.append(float(row["latencia_ms"]))
                except (ValueError, KeyError):
                    continue

        if not latencias:
            print("\n📊 El log existe pero no tiene mediciones aún.")
            return

        latencias.sort()
        n: int = len(latencias)
        promedio: float = sum(latencias) / n
        p95_idx: int = max(0, int(n * 0.95) - 1)
        p95: float = latencias[p95_idx]
        cumple_p95: bool = p95 < 100.0

        print("\n" + "=" * 40)
        print("📊 RESUMEN DE LATENCIAS")
        print("=" * 40)
        print(f"  Consultas totales : {n}")
        print(f"  Promedio          : {promedio:.2f} ms")
        print(f"  Mínimo            : {latencias[0]:.2f} ms")
        print(f"  Máximo            : {latencias[-1]:.2f} ms")
        print(f"  P95               : {p95:.2f} ms  {'✅ Cumple < 100ms' if cumple_p95 else '❌ No cumple < 100ms'}")
        print("=" * 40)

    except Exception as e:
        print(f"⚠️  No se pudo leer el log de latencias: {e}")
