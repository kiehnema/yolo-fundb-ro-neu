import streamlit as st
import sqlite3
from datetime import datetime
import os
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile

# --- Konfiguration ---
st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔍",
    layout="wide"
)

# --- Datenbank Initialisierung ---
def init_db():
    conn = sqlite3.connect('lost_and_found.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS items
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_path TEXT,
                 objects TEXT,
                 item_type TEXT,
                 timestamp TEXT,
                 contact_info TEXT)''')
    conn.commit()
    return conn

conn = init_db()

# --- YOLO Modell Initialisierung ---
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        return None

model = load_model()

# --- Hilfsfunktionen ---
def save_uploaded_file(uploaded_file):
    """Speichert hochgeladene Datei und gibt Pfad zurück"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def detect_objects(image_path, model):
    """Erkennt Objekte mit YOLO und gibt DataFrame zurück"""
    results = model(image_path)
    detections = []
    
    for result in results:
        for box in result.boxes:
            detections.append({
                'name': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'xmin': float(box.xyxy[0][0]),
                'ymin': float(box.xyxy[0][1]),
                'xmax': float(box.xyxy[0][2]),
                'ymax': float(box.xyxy[0][3])
            })
    
    return pd.DataFrame(detections)

def draw_boxes_pil(image_path, detected_objects):
    """Zeichnet Bounding Boxes mit PIL"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for _, obj in detected_objects.iterrows():
        box = [
            obj['xmin'], obj['ymin'],
            obj['xmax'], obj['ymax']
        ]
        draw.rectangle(box, outline="red", width=3)
        label = f"{obj['name']} {obj['confidence']:.2f}"
        draw.text((box[0], box[1] - 15), label, fill="red")
    
    return image

# --- Streamlit UI ---
def upload_page():
    st.header("🔹 Gegenstand melden")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
        item_type = st.radio("Art des Eintrags", ["Verloren", "Gefunden"], index=0)
        contact_info = st.text_input("Kontaktinformation (optional)")
    
    with col2:
        if uploaded_file:
            st.image(uploaded_file, caption="Hochgeladenes Bild", use_column_width=True)
    
    if uploaded_file and st.button("Eintrag speichern"):
        with st.spinner("Analysiere Bild..."):
            try:
                image_path = save_uploaded_file(uploaded_file)
                detections = detect_objects(image_path, model)
                
                if not detections.empty:
                    result_image = draw_boxes_pil(image_path, detections)
                    st.image(result_image, caption="Erkannte Objekte", use_column_width=True)
                    
                    objects_str = ", ".join(detections['name'].unique())
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    conn.execute(
                        "INSERT INTO items (image_path, objects, item_type, timestamp, contact_info) VALUES (?, ?, ?, ?, ?)",
                        (image_path, objects_str, item_type, timestamp, contact_info)
                    )
                    conn.commit()
                    
                    st.success("✅ Eintrag erfolgreich gespeichert!")
                    st.json({
                        "Erkannte Objekte": list(detections['name'].unique()),
                        "Typ": item_type,
                        "Zeitpunkt": timestamp
                    })
                else:
                    st.warning("⚠️ Keine Objekte erkannt. Bitte besseres Foto versuchen.")
            
            except Exception as e:
                st.error(f"Fehler: {str(e)}")

def search_page():
    st.header("🔍 Gegenstände suchen")
    
    search_query = st.text_input("Suchbegriff eingeben (z.B. 'Schlüssel', 'Handy')")
    item_filter = st.selectbox("Filter nach Typ", ["Alle", "Verloren", "Gefunden"])
    
    query = "SELECT * FROM items"
    conditions = []
    
    if search_query:
        conditions.append(f"objects LIKE '%{search_query}%'")
    if item_filter != "Alle":
        conditions.append(f"item_type = '{item_filter}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    results = conn.execute(query).fetchall()
    
    if results:
        st.subheader(f"Ergebnisse ({len(results)})")
        
        for item in results:
            with st.expander(f"🔸 {item[3]} - {item[2]} ({item[4]})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    try:
                        st.image(item[1], use_column_width=True)
                    except:
                        st.warning("Bild konnte nicht geladen werden")
                
                with col2:
                    st.write(f"**Typ:** {item[3]}")
                    st.write(f"**Objekte:** {item[2]}")
                    st.write(f"**Datum:** {item[4]}")
                    if item[5]:
                        st.write(f"**Kontakt:** {item[5]}")
    else:
        st.info("Keine Ergebnisse gefunden")

def browse_page():
    st.header("📋 Alle Einträge")
    
    all_items = conn.execute("SELECT * FROM items ORDER BY timestamp DESC").fetchall()
    
    if all_items:
        for item in all_items:
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                try:
                    st.image(item[1], use_column_width=True)
                except:
                    st.warning("Bild nicht verfügbar")
            
            with col2:
                st.subheader(f"{item[3]} - {item[2]}")
                st.caption(f"Eingetragen am: {item[4]}")
                if item[5]:
                    st.write(f"**Kontakt:** {item[5]}")
                
                if st.button("Details anzeigen", key=f"btn_{item[0]}"):
                    detections = pd.DataFrame([{'name': obj} for obj in item[2].split(", ")])
                    st.write("Erkannte Objekte:")
                    st.dataframe(detections['name'].value_counts())
    else:
        st.info("Noch keine Einträge vorhanden")

# --- Hauptnavigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Menü",
    ["Gegenstand melden", "Gegenstände suchen", "Alle Einträge"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    """Digitales Fundbüro v1.0

Funktionen:
- Objekterkennung mit YOLOv8
- Verloren/Gefunden-Meldungen
- Durchsuchbare Datenbank"""
)

if page == "Gegenstand melden":
    upload_page()
elif page == "Gegenstände suchen":
    search_page()
elif page == "Alle Einträge":
    browse_page()

# --- Aufräumen ---
conn.close()
