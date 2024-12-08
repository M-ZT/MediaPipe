import mediapipe as mp
import cv2

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializa la herramienta de dibujo de MediaPipe
mp_drawing = mp.solutions.drawing_utils

# Carga la imagen
image = cv2.imread('testimg.jpg')  # Cambia esto por el path de tu imagen
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte la imagen a RGB

# Realiza la detección de poses
results = pose.process(image_rgb)

# Si se detecta una pose, dibuja los puntos de la pose sobre la imagen original
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Guarda la imagen con los puntos de la pose detectados
cv2.imwrite('pose_detected_image.jpg', image)

# Imprime el nombre del archivo guardado
print("Imagen guardada como 'pose_detected_image.jpg'")
