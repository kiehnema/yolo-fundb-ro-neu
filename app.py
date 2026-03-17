import streamlit as st
import sqlite3
from datetime import datetime
from PIL import Image, ImageDraw
import pandas as pd
from ultralytics import YOLO
import tempfile
import numpy as np

# --- Konfiguration ---
st.set_page_config(
    page_title="Digitales Fundbüro",
    page_icon="🔍",
    layout="wide"
)

# --- Datenbank Initialisierung ---
def init_db():
    conn = sqlite3.connect('lost_and_found.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS items
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_path TEXT,
                 objects TEXT,
                 item_type TEXT,
                 timestamp TEXT,
                 contact_info TEXT,
                 location TEXT)''')
    conn.commit()
    return conn

conn = init_db()

# --- Modell Initialisierung mit Caching ---
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        st.stop()

model = load_model()

# --- Hilfsfunktionen ---
def save_uploaded_file(uploaded_file):
    """Sicherer Datei-Upload mit tempfile"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Datei konnte nicht gespeichert werden: {e}")
        return None

def detect_objects(image_path):
    """Objekterkennung mit Fehlerbehandlung"""
    try:
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
    except Exception as e:
        st.error(f"Objekterkennung fehlgeschlagen: {e}")
        return pd.DataFrame()

def draw_boxes(image_path, detections):
    """Pillow-basierte Box-Zeichnung"""
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        for _, obj in detections.iterrows():
            box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]-15), 
                     f"{obj['name']} {obj['confidence']:.2f}", 
                     fill="red")
        return image
    except Exception as e:
        st.error(f"Bildverarbeitung fehlgeschlagen: {e}")
        return Image.open(image_path)

# --- UI-Komponenten ---
def upload_page():
    st.header("🔹 Gegenstand melden")
    
    with st.form("upload_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
            item_type = st.radio("Art des Eintrags", ["Verloren", "Gefunden"], index=0)
            location = st.text_input("Fundort/Verlustort")
            
        with col2:
            if uploaded_file:
                st.image(uploaded_file, caption="Hochgeladenes Bild", use_column_width=True)
        
        contact_info = st.text_input("Kontaktinformationen")
        submit_button = st.form_submit_button("Eintrag speichern")
    
    if submit_button and uploaded_file:
        with st.spinner("Analysiere Bild..."):
            try:
                image_path = save_uploaded_file(uploaded_file)
                if not image_path:
                    return
                
                detections = detect_objects(image_path)
                
                if not detections.empty:
                    result_image = draw_boxes(image_path, detections)
                    st.image(result_image, caption="Erkannte Objekte", use_column_width=True)
                    
                    objects_str = ", ".join(detections['name'].unique())
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    conn.execute(
                        """INSERT INTO items 
                        (image_path, objects, item_type, timestamp, contact_info, location) 
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (image_path, objects_str, item_type, timestamp, contact_info, location)
                    )
                    conn.commit()
                    
                    st.success("✅ Eintrag erfolgreich gespeichert!")
                    st.json({
                        "Erkannte Objekte": list(detections['name'].unique()),
                        "Typ": item_type,
                        "Ort": location,
                        "Zeitpunkt": timestamp
                    })
                else:
                    st.warning("⚠️ Keine Objekte erkannt. Bitte besseres Foto versuchen.")
            
            except Exception as e:
                st.error(f"Fehler beim Speichern: {str(e)}")

def search_page():
    st.header("🔍 Gegenstände suchen")
    
    with st.expander("Suchfilter", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Suchbegriff (z.B. 'Schlüssel', 'Handy')")
        with col2:
            item_filter = st.selectbox("Typ", ["Alle", "Verloren", "Gefunden"])
        location_filter = st.text_input("Standort (optional)")
    
    query = "SELECT * FROM items"
    conditions = []
    
    if search_query:
        conditions.append(f"objects LIKE '%{search_query}%'")
    if item_filter != "Alle":
        conditions.append(f"item_type = '{item_filter}'")
    if location_filter:
        conditions.append(f"location LIKE '%{location_filter}%'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC"
    results = conn.execute(query).fetchall()
    
    if results:
        st.subheader(f"Ergebnisse ({len(results)})")
        for item in results:
            with st.expander(f"{item[3]} - {item[2]} ({item[4]})"):
                display_item(item)
    else:
        st.info("Keine Ergebnisse gefunden")

def display_item(item):
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
        if item[6]:
            st.write(f"**Ort:** {item[6]}")
        if item[5]:
            st.write(f"**Kontakt:** {item[5]}")
        
        if st.button("Details", key=f"details_{item[0]}"):
            detections = pd.DataFrame([{'name': obj} for obj in item[2].split(", ")])
            st.write("Erkannte Objekte:")
            st.dataframe(detections['name'].value_counts().rename("Anzahl"))

# --- Hauptnavigation ---
st.sidebar.title("Digitales Fundbüro")
page = st.sidebar.radio(
    "Menü",
    ["Gegenstand melden", "Gegenstände suchen"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Funktionen:**
- 🖼️ Bild-Upload mit Objekterkennung
- 🔍 Erweiterte Suchfilter
- 📍 Standortinformationen
- 📊 Statistische Auswertung
""")

if page == "Gegenstand melden":
    upload_page()
elif page == "Gegenstände suchen":
    search_page()

# --- Aufräumen ---
conn.close()
