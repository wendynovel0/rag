"""
Módulo de inicialización de la base de datos SQLite local.

Crea las tablas necesarias para el sistema FitIA si no existen:
- usuarios: perfil y objetivo del usuario
- registro_comidas: log diario de alimentos consumidos
- registro_ejercicios: log diario de ejercicios realizados
- planes_semanales: planes alimentarios y de entrenamiento guardados
"""

import sqlite3
import os
from datetime import datetime

DB_PATH: str = "fitia.db"


def get_connection() -> sqlite3.Connection:
    """
    Abre y devuelve una conexión a la base de datos SQLite local.

    Configura row_factory para que los resultados se puedan acceder
    como diccionarios en lugar de tuplas planas.

    Returns:
        sqlite3.Connection: Conexión activa a la base de datos.

    Raises:
        RuntimeError: Si no se puede abrir o crear el archivo de base de datos.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise RuntimeError(f"No se pudo conectar a la base de datos '{DB_PATH}': {e}") from e


def init_db() -> None:
    """
    Inicializa la base de datos creando todas las tablas si no existen.

    Es seguro llamar esta función múltiples veces: usa CREATE TABLE IF NOT EXISTS
    para no destruir datos existentes en ejecuciones anteriores.

    Raises:
        RuntimeError: Si ocurre un error al crear las tablas.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre      TEXT    NOT NULL,
                edad        INTEGER,
                peso_kg     REAL,
                altura_cm   REAL,
                objetivo    TEXT,
                creado_en   TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS registro_comidas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id      INTEGER NOT NULL,
                alimento        TEXT    NOT NULL,
                cantidad_g      REAL    NOT NULL,
                calorias        REAL    NOT NULL,
                proteinas_g     REAL    DEFAULT 0,
                carbohidratos_g REAL    DEFAULT 0,
                grasas_g        REAL    DEFAULT 0,
                fecha           TEXT    DEFAULT (date('now')),
                hora            TEXT    DEFAULT (time('now'))
            );

            CREATE TABLE IF NOT EXISTS registro_ejercicios (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id      INTEGER NOT NULL,
                ejercicio       TEXT    NOT NULL,
                series          INTEGER DEFAULT 0,
                repeticiones    INTEGER DEFAULT 0,
                duracion_min    REAL    DEFAULT 0,
                calorias_quemadas REAL  DEFAULT 0,
                fecha           TEXT    DEFAULT (date('now'))
            );

            CREATE TABLE IF NOT EXISTS planes_semanales (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id  INTEGER NOT NULL,
                tipo        TEXT    NOT NULL,
                contenido   TEXT    NOT NULL,
                creado_en   TEXT    DEFAULT (datetime('now'))
            );
        """)

        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error al inicializar las tablas de la base de datos: {e}") from e
