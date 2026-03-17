# digital_fundbuero_yolo.py
"""
Digitale Fundbüro App – echte YOLO-Objekterkennung, CPU-only, Streamlit Cloud-kompatibel
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import sqlite3
import os
from datetime import datetime
import torch

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
# 2. YOLO Modell laden (CPU-only)
# ---------------------------
@st.cache_resource
def load_model():
    # YOLOv8n CPU-only
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # kleines, schnelles Modell
    return model

model = load_model()

# ---------------------------
# 3. Helper Funktionen
# ---------------------------
def save_image(image: Image.Image):
    temp_dir = "images"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(temp_dir, f"{timestamp}.png")
    image.save(path)
    return path

def detect_objects(image_path: str):
    # YOLO erkennt Objekte
    results = model.predict(source=image_path, verbose=False)
    objs = []
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            objs.append(f"{label} ({conf:.2f})")
            # Bounding Box zeichnen
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1-10), f"{label} {conf:.2f}", fill="green", font=font)
    
    boxed_path = image_path.replace(".png", "_boxed.png")
    img.save(boxed_path)
    return objs, boxed_path

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
# 4. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Digitales Fundbüro", layout="wide")
st.title("📦 Digitales Fundbüro (mit YOLO-Erkennung)")
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
        
        if st.button("Erkennen & Speichern"):
            image_path = save_image(image)
            objects, boxed_path = detect_objects(image_path)
            if objects:
                insert_item(image_path, objects, item_type)
                st.success(f"{len(objects)} Objekte erkannt und gespeichert.")
                st.image(boxed_path, caption="Erkannte Objekte", use_column_width=True)
                st.write(objects)
            else:
                st.warning("Keine Objekte erkannt.")

# ----- Suche & Übersicht -----
with tab2:
    st.header("Gegenstände durchsuchen")
    search_label = st.text_input("Nach Objektlabel suchen (optional)")

    if st.button("Suchen") or st.button("Alle Gegenstände anzeigen"):
        results = query_items(search_label if search_label else None)
        if results:
            for row in results:
                st.subheader(f"ID: {row[0]} | {row[3]} | {row[4][:19]}")
                st.image(row[1].replace(".png", "_boxed.png"), width=300)
                st.write("Erkannte Objekte:", row[2])
        else:
            st.info("Keine Gegenstände gefunden.")
