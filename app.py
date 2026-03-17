# digital_fundbuero.py
"""
Digitale Fundbüro Streamlit-App
--------------------------------
Funktionen:
- Upload von Bildern verlorener oder gefundener Gegenstände
- Automatische Objekterkennung mit YOLOv8
- Anzeige von Bounding Boxes auf dem Originalbild
- Speicherung in SQLite-Datenbank
- Suchfunktion nach Objektlabels
- Übersicht aller Gegenstände

Deployment:
1. Stelle sicher, dass Streamlit und alle dependencies installiert sind.
2. Deploy auf https://streamlit.io/cloud
"""

import streamlit as st
from PIL import Image
import sqlite3
import io
import os
from datetime import datetime
import pandas as pd
import tempfile

# YOLO Import (Ultralytics YOLOv8)
from ultralytics import YOLO
import cv2
import numpy as np

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
# 2. YOLO Setup
# ---------------------------
@st.cache_resource
def load_model():
    # YOLOv8 small model, erkennt viele Objekte
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------
# 3. Helper Funktionen
# ---------------------------

def save_image(image: Image.Image):
    """Speichert Bild temporär und gibt Pfad zurück"""
    temp_dir = "images"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(temp_dir, f"{timestamp}.png")
    image.save(path)
    return path

def detect_objects(image_path: str):
    """Erkennt Objekte im Bild mit YOLOv8"""
    results = model.predict(image_path)
    detected_objects = []
    # Erstelle Kopie für Bounding Boxen
    image = cv2.imread(image_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            detected_objects.append(f"{label} ({conf:.2f})")
            # Bounding Box zeichnen
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    # Speichern mit Bounding Boxes
    boxed_path = image_path.replace(".png", "_boxed.png")
    cv2.imwrite(boxed_path, image)
    return detected_objects, boxed_path

def insert_item(image_path, objects, item_type):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO items (image_path, objects, item_type, timestamp) VALUES (?, ?, ?, ?)",
              (image_path, ", ".join(objects), item_type, datetime.now().isoformat()))
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
# 4. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Digitales Fundbüro", layout="wide")

st.title("📦 Digitales Fundbüro")
st.markdown("Finde oder melde verlorene Gegenstände.")

# Tabs für Upload und Suche
tab1, tab2 = st.tabs(["Gegenstand melden", "Gegenstände durchsuchen"])

with tab1:
    st.header("Neuen Gegenstand hinzufügen")
    uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
    item_type = st.radio("Typ des Gegenstands:", ["verloren", "gefunden"])
    if uploaded_file and st.button("Hochladen & Objekte erkennen"):
        image = Image.open(uploaded_file).convert("RGB")
        image_path = save_image(image)
        objects, boxed_path = detect_objects(image_path)
        insert_item(image_path, objects, item_type)
        st.success(f"Erkennung abgeschlossen: {len(objects)} Objekte gefunden")
        st.image(boxed_path, caption="Erkannte Objekte", use_column_width=True)
        st.write(objects)

with tab2:
    st.header("Gegenstände durchsuchen")
    search_label = st.text_input("Nach Objektlabel suchen")
    if st.button("Suchen") or st.button("Alle Gegenstände anzeigen"):
        results = query_items(search_label if search_label else None)
        if results:
            for row in results:
                st.subheader(f"ID: {row[0]} | {row[3]} | {row[4][:19]}")
                st.image(row[1].replace(".png", "_boxed.png"), width=300)
                st.write("Erkannte Objekte:", row[2])
        else:
            st.info("Keine Gegenstände gefunden.")

st.markdown("---")
st.markdown("© 2026 Digitales Fundbüro")
