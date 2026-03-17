import streamlit as st
import sqlite3
from datetime import datetime
import os
from PIL import Image
import cv2
import torch
from yoloworld import YOLOWorld  # YOLO-World implementation

# Initialize YOLO-World model
model = YOLOWorld(model_name='yolo_world')

# Initialize SQLite database
conn = sqlite3.connect('lost_and_found.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS items
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              image_path TEXT,
              objects TEXT,
              item_type TEXT,
              timestamp TEXT)''')
conn.commit()

# Streamlit App Layout
st.title("Digitales Fundbüro")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite auswählen", ["Gegenstand hochladen", "Gegenstände durchsuchen", "Alle Gegenstände anzeigen"])

# Function to save uploaded image
def save_uploaded_file(uploaded_file):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to detect objects in an image using YOLO-World
def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    detected_objects = results.pandas().xyxy[0]  # Get detected objects
    return detected_objects

# Function to draw bounding boxes on the image
def draw_boxes(image_path, detected_objects):
    image = cv2.imread(image_path)
    for _, obj in detected_objects.iterrows():
        label = obj['name']
        confidence = obj['confidence']
        x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Page: Upload Item
if page == "Gegenstand hochladen":
    st.header("Gegenstand hochladen")
    uploaded_file = st.file_uploader("Bild des Gegenstands hochladen", type=["jpg", "jpeg", "png"])
    item_type = st.radio("Typ des Gegenstands", ["verloren", "gefunden"])

    if uploaded_file is not None and item_type:
        file_path = save_uploaded_file(uploaded_file)
        detected_objects = detect_objects(file_path)
        image_with_boxes = draw_boxes(file_path, detected_objects)
        st.image(image_with_boxes, caption="Erkannte Objekte", use_column_width=True)

        # Save to database
        objects = ", ".join(detected_objects['name'].tolist())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO items (image_path, objects, item_type, timestamp) VALUES (?, ?, ?, ?)",
                  (file_path, objects, item_type, timestamp))
        conn.commit()
        st.success("Gegenstand erfolgreich hochgeladen und gespeichert!")

# Page: Search Items
elif page == "Gegenstände durchsuchen":
    st.header("Gegenstände durchsuchen")
    search_query = st.text_input("Nach Objektlabels suchen")
    if search_query:
        c.execute("SELECT * FROM items WHERE objects LIKE ?", (f"%{search_query}%",))
        results = c.fetchall()
        if results:
            st.write("Gefundene Gegenstände:")
            for result in results:
                st.write(f"ID: {result[0]}, Typ: {result[3]}, Zeitstempel: {result[4]}")
                st.image(result[1], caption=result[2], use_column_width=True)
        else:
            st.warning("Keine Gegenstände gefunden.")

# Page: Show All Items
elif page == "Alle Gegenstände anzeigen":
    st.header("Alle Gegenstände")
    c.execute("SELECT * FROM items")
    all_items = c.fetchall()
    if all_items:
        for item in all_items:
            st.write(f"ID: {item[0]}, Typ: {item[3]}, Zeitstempel: {item[4]}")
            st.image(item[1], caption=item[2], use_column_width=True)
    else:
        st.info("Keine Gegenstände vorhanden.")

# Close database connection
conn.close()
