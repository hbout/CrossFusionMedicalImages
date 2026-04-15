import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Charger l'image
img_path = 'test_results/7.png'  # Ajuste ce chemin
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image non trouvée à : {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# Définir le petit carré rouge (position centrale comme dans ton exemple)
box_x = 170  # position horizontale depuis la gauche
box_y = 60   # position verticale depuis le haut
box_width = 25  # Réduit à 30 pixels
box_height = 25  # Réduit à 30 pixels
square_top_left = (box_x, box_y)
square_bottom_right = (box_x + box_width, box_y + box_height)

# Créer la fenêtre Tkinter
root = tk.Tk()
root.title("Image avec carré rouge et zoom")

# Convertir l'image en ImageTk
def get_tk_image(image):
    return ImageTk.PhotoImage(Image.fromarray(image))

# Ajouter le petit carré rouge
img_display = img.copy()
cv2.rectangle(img_display, square_top_left, square_bottom_right, (255, 0, 0), 2)

# Préparer le zoom dans le coin inférieur droit
zoom_size = 90  # Restauré à 60 pixels comme dans la version initiale
zoom_top_left = (w - zoom_size - 2, h - zoom_size - 2)
zoom_bottom_right = (w - 2, h - 2)

canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()
tk_img = get_tk_image(img_display)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# Variable pour suivre si le carré est en train d'être déplacé
dragging = False
offset_x = 0
offset_y = 0

# Fonction pour mettre à jour le zoom avec la zone du carré rouge
def update_zoom():
    global tk_img
    # Extraire la zone du petit carré rouge
    zoom_area = img[max(0, box_y):min(h, box_y + box_height), max(0, box_x):min(w, box_x + box_width)]
    if zoom_area.size != 0:
        # Redimensionner la zone pour correspondre exactement à la taille du zoom
        zoom_img = cv2.resize(zoom_area, (zoom_size - 4, zoom_size - 4), interpolation=cv2.INTER_NEAREST)
        # Appliquer le zoom dans le coin inférieur droit avec padding si nécessaire
        img_with_zoom = img_display.copy()
        y_start = zoom_top_left[1]
        y_end = zoom_bottom_right[1]
        x_start = zoom_top_left[0]
        x_end = zoom_bottom_right[0]
        h_zoom, w_zoom, _ = zoom_img.shape
        img_with_zoom[y_start:y_start + h_zoom, x_start:x_start + w_zoom] = zoom_img
        # Ajouter un cadre rouge autour de la zone de zoom
        cv2.rectangle(img_with_zoom, zoom_top_left, zoom_bottom_right, (255, 0, 0), 2)
        tk_img = get_tk_image(img_with_zoom)
        canvas.itemconfig(image_on_canvas, image=tk_img)

# Fonction pour démarrer le déplacement
def start_drag(event):
    global dragging, offset_x, offset_y
    x, y = event.x, event.y
    if (box_x <= x <= box_x + box_width and box_y <= y <= box_y + box_height):
        dragging = True
        offset_x = x - box_x
        offset_y = y - box_y

# Fonction pour déplacer le carré
def drag(event):
    global box_x, box_y, img_display, tk_img
    if dragging:
        new_x = max(0, min(w - box_width, event.x - offset_x))
        new_y = max(0, min(h - box_height, event.y - offset_y))
        box_x, box_y = new_x, new_y
        # Mettre à jour l'image avec le nouveau carré
        img_display = img.copy()
        cv2.rectangle(img_display, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 0, 0), 2)
        update_zoom()

# Fonction pour arrêter le déplacement
def stop_drag(event):
    global dragging
    dragging = False

# Lier les événements de la souris
canvas.bind("<Button-1>", start_drag)
canvas.bind("<B1-Motion>", drag)
canvas.bind("<ButtonRelease-1>", stop_drag)

# Mettre à jour le zoom initial
update_zoom()

root.mainloop()