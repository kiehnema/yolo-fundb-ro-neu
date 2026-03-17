# fundbuero_app.py
import streamlit as st
import sqlite3
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
import torch

# ---------------------------
# Setup SQLite
# ---------------------------
DB_PATH = "fundbuero.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    labels TEXT,
    status TEXT,
    timestamp TEXT
)
''')
conn.commit()

# ---------------------------
# Load YOLO model (YOLOv8 as alternative)
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # YOLOv5 als Alternative
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des YOLO-Modells: {e}")
        return None

model = load_model()

# ---------------------------
# Helper functions
# ---------------------------
def save_image(uploaded_file):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    image_path = upload_dir / uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(image_path)

def detect_objects(image_path):
    img = Image.open(image_path)
    results = model(img)
    labels = []
    draw = ImageDraw.Draw(img)
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = results.names[int(cls)]
        labels.append(label)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{label} {conf:.2f}", fill="red")
    detected_image_path = image_path.replace(".","_detected.")
    img.save(detected_image_path)
    return labels, detected_image_path

def insert_item(image_path, labels, status):
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO items (image_path, labels, status, timestamp) VALUES (?,?,?,?)",
              (image_path, ",".join(labels), status, timestamp))
    conn.commit()

def get_all_items():
    c.execute("SELECT * FROM items ORDER BY timestamp DESC")
    return c.fetchall()

def search_items(query):
    c.execute("SELECT * FROM items WHERE labels LIKE ? ORDER BY timestamp DESC", (f"%{query}%",))
    return c.fetchall()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Digitales Fundbüro", layout="wide")
st.title("📦 Digitales Fundbüro")

menu = ["Upload", "Alle Gegenstände", "Suche"]
choice = st.sidebar.selectbox("Menü", menu)

if choice == "Upload":
    st.header("Neuen Gegenstand hochladen")
    uploaded_file = st.file_uploader("Bild auswählen", type=["jpg","png","jpeg"])
    status = st.radio("Status auswählen", ["verloren", "gefunden"])
    
    if uploaded_file and st.button("Hochladen und analysieren"):
        image_path = save_image(uploaded_file)
        labels, detected_image_path = detect_objects(image_path)
        insert_item(image_path, labels, status)
        st.success("Gegenstand erfolgreich gespeichert!")
        st.image(detected_image_path, caption="Erkanntes Bild mit Bounding Boxes", use_column_width=True)
        st.write("Erkannte Objekte:", labels)

elif choice == "Alle Gegenstände":
    st.header("Alle Gegenstände")
    items = get_all_items()
    for item in items:
        st.image(item[1], use_column_width=True)
        st.write(f"Labels: {item[2]}, Status: {item[3]}, Hochgeladen am: {item[4]}")

elif choice == "Suche":
    st.header("Suche nach Gegenständen")
    query = st.text_input("Objektname eingeben")
    if query:
        results = search_items(query)
        if results:
            for item in results:
                st.image(item[1], use_column_width=True)
                st.write(f"Labels: {item[2]}, Status: {item[3]}, Hochgeladen am: {item[4]}")
        else:
            st.info("Keine Gegenstände gefunden.")
