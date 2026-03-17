# digital_fundbuero_app.py
import streamlit as st
from PIL import Image
import sqlite3
from datetime import datetime
import os
import torch
import tempfile
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
# 2. YOLO Modell (CPU-only, stabil auf Streamlit Cloud)
# ---------------------------
@st.cache_resource
def load_model():
    # YOLOv5s über Torch Hub, CPU-only
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.cpu()
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
    img = Image.open(image_path)
    results = model(img, size=640)
    objects = []
    # Bounding Boxes auf Bild zeichnen
    im = results.render()[0]  # render() gibt numpy array
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        objects.append(f"{label} ({conf:.2f})")
    # Speichern mit Bounding Boxes
    boxed_path = image_path.replace(".png", "_boxed.png")
    Image.fromarray(im).save(boxed_path)
    return objects, boxed_path

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
        st.success(f"{len(objects)} Objekte erkannt")
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
