"""
Catálogo de funciones locales para el sistema FitIA — Fase 2: Function Calling.

Contiene las 10 funciones que el LLM puede invocar mediante JSON estructurado.
Cada función está completamente tipada y documentada para que el modelo pueda
entender cuándo y cómo usarla a partir de sus docstrings y type hints.

Funciones disponibles:
    1.  crear_usuario            — Registrar perfil del usuario
    2.  obtener_perfil           — Consultar datos del usuario
    3.  registrar_comida         — Agregar alimento al log diario
    4.  consultar_calorias_hoy   — Ver ingesta calórica del día
    5.  eliminar_ultima_comida   — Borrar el último alimento registrado
    6.  registrar_ejercicio      — Agregar ejercicio al log diario
    7.  consultar_ejercicios_hoy — Ver ejercicios del día
    8.  calcular_balance_calorico— Calcular diferencia ingesta vs gasto
    9.  guardar_plan_semanal     — Guardar un plan alimentario o de entrenamiento
    10. consultar_historial      — Ver resumen de los últimos N días
"""

import sqlite3
from datetime import date, timedelta
from src.database import get_connection


# ---------------------------------------------------------------------------
# 1. CREAR USUARIO
# ---------------------------------------------------------------------------

def crear_usuario(
    nombre: str,
    edad: int,
    peso_kg: float,
    altura_cm: float,
    objetivo: str
) -> dict:
    """
    Registra un nuevo usuario con su perfil y objetivo de salud.

    Crea el perfil base del usuario en la base de datos. Si ya existe un usuario
    con el mismo nombre, devuelve un error en lugar de crear un duplicado.

    Args:
        nombre      (str):   Nombre completo del usuario.
        edad        (int):   Edad en años.
        peso_kg     (float): Peso actual en kilogramos.
        altura_cm   (float): Altura en centímetros.
        objetivo    (str):   Meta de salud. Valores válidos:
                             "bajar peso", "subir peso", "mantener peso",
                             "ganar músculo", "mejorar resistencia".

    Returns:
        dict: {"ok": True, "usuario_id": int, "mensaje": str} si fue exitoso.
              {"ok": False, "error": str} si ocurrió un problema.
    """
    objetivos_validos = {"bajar peso", "subir peso", "mantener peso", "ganar músculo", "mejorar resistencia"}
    if objetivo.lower() not in objetivos_validos:
        return {"ok": False, "error": f"Objetivo no válido. Opciones: {', '.join(objetivos_validos)}"}

    if edad <= 0 or edad > 120:
        return {"ok": False, "error": "La edad debe estar entre 1 y 120 años."}

    if peso_kg <= 0 or peso_kg > 500:
        return {"ok": False, "error": "El peso debe estar entre 1 y 500 kg."}

    if altura_cm <= 0 or altura_cm > 300:
        return {"ok": False, "error": "La altura debe estar entre 1 y 300 cm."}

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Verificar duplicado por nombre
        existente = cursor.execute(
            "SELECT id FROM usuarios WHERE nombre = ?", (nombre,)
        ).fetchone()

        if existente:
            conn.close()
            return {"ok": False, "error": f"Ya existe un usuario con el nombre '{nombre}'. Usa otro nombre o consulta tu perfil."}

        cursor.execute(
            "INSERT INTO usuarios (nombre, edad, peso_kg, altura_cm, objetivo) VALUES (?, ?, ?, ?, ?)",
            (nombre, edad, peso_kg, altura_cm, objetivo.lower())
        )
        conn.commit()
        usuario_id = cursor.lastrowid
        conn.close()

        return {"ok": True, "usuario_id": usuario_id, "mensaje": f"Perfil de '{nombre}' creado correctamente."}

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al crear usuario: {e}"}


# ---------------------------------------------------------------------------
# 2. OBTENER PERFIL
# ---------------------------------------------------------------------------

def obtener_perfil(usuario_id: int) -> dict:
    """
    Devuelve el perfil completo de un usuario registrado.

    Args:
        usuario_id (int): ID numérico del usuario (obtenido al crear el perfil).

    Returns:
        dict: {"ok": True, "perfil": dict} con nombre, edad, peso, altura y objetivo.
              {"ok": False, "error": str} si el ID no existe.
    """
    if usuario_id <= 0:
        return {"ok": False, "error": "El ID de usuario debe ser un número positivo."}

    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM usuarios WHERE id = ?", (usuario_id,)
        ).fetchone()
        conn.close()

        if not row:
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        return {
            "ok": True,
            "perfil": {
                "id": row["id"],
                "nombre": row["nombre"],
                "edad": row["edad"],
                "peso_kg": row["peso_kg"],
                "altura_cm": row["altura_cm"],
                "objetivo": row["objetivo"],
                "creado_en": row["creado_en"]
            }
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al obtener perfil: {e}"}


# ---------------------------------------------------------------------------
# 3. REGISTRAR COMIDA
# ---------------------------------------------------------------------------

def registrar_comida(
    usuario_id: int,
    alimento: str,
    cantidad_g: float,
    calorias: float,
    proteinas_g: float = 0.0,
    carbohidratos_g: float = 0.0,
    grasas_g: float = 0.0
) -> dict:
    """
    Registra un alimento consumido en el log diario del usuario.

    Guarda el alimento con su información nutricional en la fecha actual.
    Las calorías son obligatorias; los macronutrientes son opcionales pero
    recomendados para un seguimiento más completo.

    Args:
        usuario_id      (int):   ID del usuario que consumió el alimento.
        alimento        (str):   Nombre del alimento (ej: "arroz blanco", "pechuga de pollo").
        cantidad_g      (float): Cantidad consumida en gramos.
        calorias        (float): Calorías totales de la porción indicada.
        proteinas_g     (float): Gramos de proteína. Por defecto 0.
        carbohidratos_g (float): Gramos de carbohidratos. Por defecto 0.
        grasas_g        (float): Gramos de grasa. Por defecto 0.

    Returns:
        dict: {"ok": True, "registro_id": int, "mensaje": str} si fue exitoso.
              {"ok": False, "error": str} si ocurrió un problema.
    """
    if cantidad_g <= 0:
        return {"ok": False, "error": "La cantidad debe ser mayor a 0 gramos."}

    if calorias < 0:
        return {"ok": False, "error": "Las calorías no pueden ser negativas."}

    if not alimento.strip():
        return {"ok": False, "error": "El nombre del alimento no puede estar vacío."}

    try:
        conn = get_connection()

        # Verificar que el usuario existe
        usuario = conn.execute("SELECT id FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        if not usuario:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        cursor = conn.execute(
            """INSERT INTO registro_comidas
               (usuario_id, alimento, cantidad_g, calorias, proteinas_g, carbohidratos_g, grasas_g)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (usuario_id, alimento.strip(), cantidad_g, calorias, proteinas_g, carbohidratos_g, grasas_g)
        )
        conn.commit()
        registro_id = cursor.lastrowid
        conn.close()

        return {
            "ok": True,
            "registro_id": registro_id,
            "mensaje": f"'{alimento}' registrado correctamente ({calorias} kcal)."
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al registrar comida: {e}"}


# ---------------------------------------------------------------------------
# 4. CONSULTAR CALORÍAS HOY
# ---------------------------------------------------------------------------

def consultar_calorias_hoy(usuario_id: int) -> dict:
    """
    Devuelve el resumen de calorías e ingesta nutricional del día actual del usuario.

    Suma todas las comidas registradas hoy y las devuelve con el desglose
    por macronutrientes (proteínas, carbohidratos, grasas).

    Args:
        usuario_id (int): ID del usuario a consultar.

    Returns:
        dict: {"ok": True, "fecha": str, "total_calorias": float,
               "total_proteinas_g": float, "total_carbohidratos_g": float,
               "total_grasas_g": float, "comidas": list[dict]} si fue exitoso.
              {"ok": False, "error": str} si el usuario no existe.
    """
    if usuario_id <= 0:
        return {"ok": False, "error": "El ID de usuario debe ser un número positivo."}

    try:
        conn = get_connection()
        hoy = date.today().isoformat()

        usuario = conn.execute("SELECT id FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        if not usuario:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        rows = conn.execute(
            """SELECT alimento, cantidad_g, calorias, proteinas_g, carbohidratos_g, grasas_g, hora
               FROM registro_comidas
               WHERE usuario_id = ? AND fecha = ?
               ORDER BY hora""",
            (usuario_id, hoy)
        ).fetchall()
        conn.close()

        comidas = [dict(r) for r in rows]
        total_cal   = sum(c["calorias"] for c in comidas)
        total_prot  = sum(c["proteinas_g"] for c in comidas)
        total_carbs = sum(c["carbohidratos_g"] for c in comidas)
        total_gras  = sum(c["grasas_g"] for c in comidas)

        return {
            "ok": True,
            "fecha": hoy,
            "total_calorias": round(total_cal, 1),
            "total_proteinas_g": round(total_prot, 1),
            "total_carbohidratos_g": round(total_carbs, 1),
            "total_grasas_g": round(total_gras, 1),
            "comidas": comidas
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al consultar calorías: {e}"}


# ---------------------------------------------------------------------------
# 5. ELIMINAR ÚLTIMA COMIDA
# ---------------------------------------------------------------------------

def eliminar_ultima_comida(usuario_id: int) -> dict:
    """
    Elimina el registro de la última comida ingresada por el usuario hoy.

    Útil cuando el usuario se equivocó al registrar un alimento. Solo elimina
    el registro más reciente del día actual, no afecta días anteriores.

    Args:
        usuario_id (int): ID del usuario cuya última comida se quiere eliminar.

    Returns:
        dict: {"ok": True, "eliminado": str, "mensaje": str} si fue exitoso.
              {"ok": False, "error": str} si no hay registros hoy o el usuario no existe.
    """
    if usuario_id <= 0:
        return {"ok": False, "error": "El ID de usuario debe ser un número positivo."}

    try:
        conn = get_connection()
        hoy = date.today().isoformat()

        ultima = conn.execute(
            """SELECT id, alimento FROM registro_comidas
               WHERE usuario_id = ? AND fecha = ?
               ORDER BY id DESC LIMIT 1""",
            (usuario_id, hoy)
        ).fetchone()

        if not ultima:
            conn.close()
            return {"ok": False, "error": "No hay comidas registradas hoy para eliminar."}

        conn.execute("DELETE FROM registro_comidas WHERE id = ?", (ultima["id"],))
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "eliminado": ultima["alimento"],
            "mensaje": f"Se eliminó '{ultima['alimento']}' del registro de hoy."
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al eliminar comida: {e}"}


# ---------------------------------------------------------------------------
# 6. REGISTRAR EJERCICIO
# ---------------------------------------------------------------------------

def registrar_ejercicio(
    usuario_id: int,
    ejercicio: str,
    series: int = 0,
    repeticiones: int = 0,
    duracion_min: float = 0.0,
    calorias_quemadas: float = 0.0
) -> dict:
    """
    Registra un ejercicio realizado por el usuario en el log del día actual.

    Acepta ejercicios de fuerza (con series y repeticiones) o cardio (con duración).
    Al menos uno de los dos grupos de parámetros debe tener valores mayores a 0.

    Args:
        usuario_id        (int):   ID del usuario que realizó el ejercicio.
        ejercicio         (str):   Nombre del ejercicio (ej: "sentadillas", "correr", "plancha").
        series            (int):   Número de series realizadas. 0 si es cardio.
        repeticiones      (int):   Repeticiones por serie. 0 si es cardio.
        duracion_min      (float): Duración en minutos. 0 si es ejercicio de fuerza.
        calorias_quemadas (float): Calorías estimadas quemadas durante el ejercicio.

    Returns:
        dict: {"ok": True, "registro_id": int, "mensaje": str} si fue exitoso.
              {"ok": False, "error": str} si ocurrió un problema.
    """
    if not ejercicio.strip():
        return {"ok": False, "error": "El nombre del ejercicio no puede estar vacío."}

    if series == 0 and repeticiones == 0 and duracion_min == 0:
        return {"ok": False, "error": "Debes indicar series/repeticiones (fuerza) o duración en minutos (cardio)."}

    if calorias_quemadas < 0:
        return {"ok": False, "error": "Las calorías quemadas no pueden ser negativas."}

    try:
        conn = get_connection()

        usuario = conn.execute("SELECT id FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        if not usuario:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        cursor = conn.execute(
            """INSERT INTO registro_ejercicios
               (usuario_id, ejercicio, series, repeticiones, duracion_min, calorias_quemadas)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (usuario_id, ejercicio.strip(), series, repeticiones, duracion_min, calorias_quemadas)
        )
        conn.commit()
        registro_id = cursor.lastrowid
        conn.close()

        return {
            "ok": True,
            "registro_id": registro_id,
            "mensaje": f"'{ejercicio}' registrado correctamente ({calorias_quemadas} kcal quemadas)."
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al registrar ejercicio: {e}"}


# ---------------------------------------------------------------------------
# 7. CONSULTAR EJERCICIOS HOY
# ---------------------------------------------------------------------------

def consultar_ejercicios_hoy(usuario_id: int) -> dict:
    """
    Devuelve todos los ejercicios registrados por el usuario en el día actual.

    Args:
        usuario_id (int): ID del usuario a consultar.

    Returns:
        dict: {"ok": True, "fecha": str, "total_calorias_quemadas": float,
               "ejercicios": list[dict]} con el detalle de cada ejercicio.
              {"ok": False, "error": str} si el usuario no existe.
    """
    if usuario_id <= 0:
        return {"ok": False, "error": "El ID de usuario debe ser un número positivo."}

    try:
        conn = get_connection()
        hoy = date.today().isoformat()

        usuario = conn.execute("SELECT id FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        if not usuario:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        rows = conn.execute(
            """SELECT ejercicio, series, repeticiones, duracion_min, calorias_quemadas
               FROM registro_ejercicios
               WHERE usuario_id = ? AND fecha = ?
               ORDER BY id""",
            (usuario_id, hoy)
        ).fetchall()
        conn.close()

        ejercicios = [dict(r) for r in rows]
        total_quemadas = sum(e["calorias_quemadas"] for e in ejercicios)

        return {
            "ok": True,
            "fecha": hoy,
            "total_calorias_quemadas": round(total_quemadas, 1),
            "ejercicios": ejercicios
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al consultar ejercicios: {e}"}


# ---------------------------------------------------------------------------
# 8. CALCULAR BALANCE CALÓRICO
# ---------------------------------------------------------------------------

def calcular_balance_calorico(usuario_id: int) -> dict:
    """
    Calcula el balance calórico del día: calorías consumidas menos calorías quemadas.

    Un balance positivo indica superávit (útil para ganar músculo o peso).
    Un balance negativo indica déficit (útil para bajar peso).
    También calcula el TMB (Tasa Metabólica Basal) estimada con la fórmula
    Mifflin-St Jeor usando los datos del perfil del usuario.

    Args:
        usuario_id (int): ID del usuario a evaluar.

    Returns:
        dict: {"ok": True, "calorias_consumidas": float, "calorias_quemadas_ejercicio": float,
               "tmb_estimada": float, "balance": float, "interpretacion": str} si fue exitoso.
              {"ok": False, "error": str} si el usuario no existe o faltan datos de perfil.
    """
    if usuario_id <= 0:
        return {"ok": False, "error": "El ID de usuario debe ser un número positivo."}

    try:
        conn = get_connection()
        hoy = date.today().isoformat()

        perfil = conn.execute(
            "SELECT peso_kg, altura_cm, edad, objetivo FROM usuarios WHERE id = ?", (usuario_id,)
        ).fetchone()

        if not perfil:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        # TMB con Mifflin-St Jeor (fórmula neutral, sin distinción de sexo)
        tmb = (10 * perfil["peso_kg"]) + (6.25 * perfil["altura_cm"]) - (5 * perfil["edad"])

        # Calorías consumidas hoy
        row_comidas = conn.execute(
            "SELECT COALESCE(SUM(calorias), 0) as total FROM registro_comidas WHERE usuario_id = ? AND fecha = ?",
            (usuario_id, hoy)
        ).fetchone()

        # Calorías quemadas por ejercicio hoy
        row_ejercicios = conn.execute(
            "SELECT COALESCE(SUM(calorias_quemadas), 0) as total FROM registro_ejercicios WHERE usuario_id = ? AND fecha = ?",
            (usuario_id, hoy)
        ).fetchone()

        conn.close()

        consumidas = round(row_comidas["total"], 1)
        quemadas   = round(row_ejercicios["total"], 1)
        balance    = round(consumidas - quemadas, 1)

        if balance > 200:
            interpretacion = "Superávit calórico — favorable para ganar músculo o peso."
        elif balance < -200:
            interpretacion = "Déficit calórico — favorable para bajar peso."
        else:
            interpretacion = "Balance equilibrado — adecuado para mantener peso."

        return {
            "ok": True,
            "calorias_consumidas": consumidas,
            "calorias_quemadas_ejercicio": quemadas,
            "tmb_estimada": round(tmb, 1),
            "balance": balance,
            "interpretacion": interpretacion
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al calcular balance: {e}"}


# ---------------------------------------------------------------------------
# 9. GUARDAR PLAN SEMANAL
# ---------------------------------------------------------------------------

def guardar_plan_semanal(usuario_id: int, tipo: str, contenido: str) -> dict:
    """
    Guarda un plan semanal alimentario o de entrenamiento para el usuario.

    Permite al nutriólogo o entrenador (o a la IA) guardar un plan estructurado
    que el usuario pueda consultar posteriormente.

    Args:
        usuario_id (int): ID del usuario al que pertenece el plan.
        tipo       (str): Tipo de plan. Valores válidos: "alimentario", "entrenamiento".
        contenido  (str): Texto completo del plan semanal (puede incluir días, comidas,
                          ejercicios, cantidades, etc.).

    Returns:
        dict: {"ok": True, "plan_id": int, "mensaje": str} si fue exitoso.
              {"ok": False, "error": str} si ocurrió un problema.
    """
    tipos_validos = {"alimentario", "entrenamiento"}
    if tipo.lower() not in tipos_validos:
        return {"ok": False, "error": f"Tipo no válido. Opciones: {', '.join(tipos_validos)}"}

    if not contenido.strip():
        return {"ok": False, "error": "El contenido del plan no puede estar vacío."}

    try:
        conn = get_connection()

        usuario = conn.execute("SELECT id FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        if not usuario:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        cursor = conn.execute(
            "INSERT INTO planes_semanales (usuario_id, tipo, contenido) VALUES (?, ?, ?)",
            (usuario_id, tipo.lower(), contenido.strip())
        )
        conn.commit()
        plan_id = cursor.lastrowid
        conn.close()

        return {
            "ok": True,
            "plan_id": plan_id,
            "mensaje": f"Plan {tipo} guardado correctamente (ID: {plan_id})."
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al guardar plan: {e}"}


# ---------------------------------------------------------------------------
# 10. CONSULTAR HISTORIAL
# ---------------------------------------------------------------------------

def consultar_historial(usuario_id: int, dias: int = 7) -> dict:
    """
    Devuelve un resumen del historial de ingesta y ejercicio de los últimos N días.

    Para cada día muestra el total de calorías consumidas, calorías quemadas
    y el balance resultante. Útil para identificar patrones y hacer seguimiento
    del progreso del usuario a lo largo de la semana.

    Args:
        usuario_id (int): ID del usuario a consultar.
        dias       (int): Número de días hacia atrás a incluir. Por defecto 7.
                          Máximo permitido: 30 días.

    Returns:
        dict: {"ok": True, "usuario_id": int, "dias_consultados": int,
               "historial": list[dict]} con un resumen por día.
              {"ok": False, "error": str} si el usuario no existe o el parámetro es inválido.
    """
    if dias <= 0 or dias > 30:
        return {"ok": False, "error": "El número de días debe estar entre 1 y 30."}

    if usuario_id <= 0:
        return {"ok": False, "error": "El ID de usuario debe ser un número positivo."}

    try:
        conn = get_connection()

        usuario = conn.execute("SELECT id FROM usuarios WHERE id = ?", (usuario_id,)).fetchone()
        if not usuario:
            conn.close()
            return {"ok": False, "error": f"No existe ningún usuario con ID {usuario_id}."}

        historial = []
        hoy = date.today()

        for i in range(dias):
            dia = (hoy - timedelta(days=i)).isoformat()

            cal = conn.execute(
                "SELECT COALESCE(SUM(calorias), 0) as total FROM registro_comidas WHERE usuario_id = ? AND fecha = ?",
                (usuario_id, dia)
            ).fetchone()["total"]

            quemadas = conn.execute(
                "SELECT COALESCE(SUM(calorias_quemadas), 0) as total FROM registro_ejercicios WHERE usuario_id = ? AND fecha = ?",
                (usuario_id, dia)
            ).fetchone()["total"]

            historial.append({
                "fecha": dia,
                "calorias_consumidas": round(cal, 1),
                "calorias_quemadas": round(quemadas, 1),
                "balance": round(cal - quemadas, 1)
            })

        conn.close()

        return {
            "ok": True,
            "usuario_id": usuario_id,
            "dias_consultados": dias,
            "historial": historial
        }

    except sqlite3.Error as e:
        return {"ok": False, "error": f"Error de base de datos al consultar historial: {e}"}
