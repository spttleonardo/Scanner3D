import cv2
import numpy as np
import os

video_path = './assets/videos/dataset/video_035.avi'
temp_video_path = './assets/videos/dataset/video_proc.avi'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[!] Falha ao abrir: {video_path}")
    exit()

# Lê primeiro frame para seleção dos pontos
ret, frame = cap.read()
if not ret:
    print("[!] Falha ao ler o primeiro frame.")
    cap.release()
    exit()

h, w = frame.shape[:2]
center_x = w // 2

# Reduzido para exibir na tela se for grande
max_width = 1000
scale = min(1.0, max_width / w)
disp_w, disp_h = int(w * scale), int(h * scale)
scale_x = w / disp_w
scale_y = h / disp_h

frame_disp = cv2.resize(frame, (disp_w, disp_h))

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        points.append((real_x, real_y))
        print(f"Ponto {len(points)}: {real_x}, {real_y}")

cv2.namedWindow("Selecione 2 pontos")
cv2.setMouseCallback("Selecione 2 pontos", mouse_callback)

while len(points) < 2:
    temp = frame_disp.copy()
    for pt in points:
        disp_pt = (int(pt[0] / scale_x), int(pt[1] / scale_y))
        cv2.circle(temp, disp_pt, 6, (0,255,0), 2)
    cv2.imshow("Selecione 2 pontos", temp)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

if len(points) != 2:
    print("[!] Seleção incompleta.")
    cap.release()
    exit()

# Ordena por y (v)
points = sorted(points, key=lambda p: p[1])
top_y = points[0][1]
bottom_y = points[1][1]

print(f"Faixa vertical: {top_y} a {bottom_y}")
print(f"Faixa horizontal: {center_x-200} a {center_x+150}")

# Cria novo VideoWriter
out = cv2.VideoWriter(
    temp_video_path,
    cv2.VideoWriter_fourcc(*'FFV1'),
    cap.get(cv2.CAP_PROP_FPS),
    (w, h)
)

# Processa todos os frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # volta ao início
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = np.ones_like(frame, dtype=np.uint8) * 255
    mask[:top_y, :] = 0
    mask[bottom_y+1:, :] = 0
    mask[:, :center_x-100] = 0
    mask[:, center_x+100:] = 0

    frame_masked = cv2.bitwise_and(frame, mask)
    out.write(frame_masked)

out.release()
cap.release()

# Substitui o original pelo novo
os.replace(temp_video_path, video_path)

print(f"[✓] Vídeo processado e salvo em: {video_path}")
