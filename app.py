# digital_fundbuero_cloud.py
"""
Digitale Fundbüro Web-App – Streamlit Cloud-kompatibel

Funktionen:
- Upload von Bildern verlorener oder gefundener Gegenstände
- Auswahl Typ (verloren / gefunden)
- Objekte als Text eingeben (manuelle "Erkennung")
- Speicherung in SQLite
- Suche nach Objektlabels
- Übersicht aller Gegenstände
"""

import streamlit as st
from PIL import Image
import sqlite3
import os
from datetime import datetime
import pandas as pd

# ---------------------------
# 1. Datenbank Setup
# ---------------------------
DB_PATH = "fundbuero.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            objects TEXT,
            item_type TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# 2. Helper Funktionen
# ---------------------------
def save_image(image: Image.Image):
    temp_dir = "images"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(temp_dir, f"{timestamp}.png")
    image.save(path)
    return path

def insert_item(image_path, objects, item_type):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO items (image_path, objects, item_type, timestamp) VALUES (?, ?, ?, ?)",
        (image_path, ", ".join(objects), item_type, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def query_items(label_filter=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if label_filter:
        c.execute("SELECT * FROM items WHERE objects LIKE ?", (f"%{label_filter}%",))
    else:
        c.execute("SELECT * FROM items")
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------------------
# 3. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Digitales Fundbüro", layout="wide")
st.title("📦 Digitales Fundbüro")
st.markdown("Finde oder melde verlorene Gegenstände.")

tab1, tab2 = st.tabs(["Gegenstand melden", "Gegenstände durchsuchen"])

# ----- Upload & Erfassung -----
with tab1:
    st.header("Neuen Gegenstand hinzufügen")
    uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
    item_type = st.radio("Typ des Gegenstands:", ["verloren", "gefunden"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        
        # Manuelle Objekterkennung über Textfeld
        obj_input = st.text_area(
            "Gefundene Objekte (durch Komma trennen, z.B. Schlüssel, Tasche):",
            placeholder="z.B. Schlüssel, Tasche"
        )
        
        if st.button("Gegenstand speichern"):
            if not obj_input.strip():
                st.warning("Bitte mindestens ein Objekt eingeben.")
            else:
                image_path = save_image(image)
                objects = [o.strip() for o in obj_input.split(",") if o.strip()]
                insert_item(image_path, objects, item_type)
                st.success(f"Gegenstand gespeichert! {len(objects)} Objekt(e) erfasst.")

# ----- Suche & Übersicht -----
with tab2:
    st.header("Gegenstände durchsuchen")
    search_label = st.text_input("Nach Objektlabel suchen (optional)")

    if st.button("Suchen") or st.button("Alle Gegenstände anzeigen"):
        results = query_items(search_label if search_label else None)
        if results:
            for row in results:
                st.subheader(f"ID: {row[0]} | {row[3]} | {row[4][:19]}")
                st.image(row[1], width=300)
                st.write("Erkannte Objekte:", row[2])
        else:
            st.info("Keine Gegenstände gefunden.")
