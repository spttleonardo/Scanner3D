import cv2
import serial
import time
import os
import requests
import numpy as np
from requests.auth import HTTPDigestAuth

# ------------- CONFIGURAÇÕES ---------------------
PORT = 'COM3'
BAUD = 9600

camera_ip = '192.168.1.64'
username = 'admin'
password = 'egl@1234'
porta_rtsp = "554"

url_snapshot = f'http://{camera_ip}/ISAPI/Streaming/channels/101/picture'
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{porta_rtsp}/Streaming/Channels/101"

# ------------- Função para nova subpasta ---------
def get_next_folder(base_path, base_name='object_'):
    i = 0
    while True:
        folder_name = f"{base_name}{chr(97+i)}"  # 'object_a', 'object_b', ...
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        i += 1

# ------------- Inicializações ---------------------
print("serial está vindo de:", serial.__file__)
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # tempo para Arduino resetar

# Cria subpasta dentro de assets
base_dir = 'assets'
save_dir = get_next_folder(base_dir)

# Conecta ao stream da Hikvision
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Erro ao conectar ao stream RTSP.")
    ser.close()
    exit()

print(f"[✓] Subpasta criada: {save_dir}")
print("[→] Pressione 's' para capturar imagem e girar motor. 'q' para sair.\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame do stream.")
        break

    # Redimensiona para visualização e mostra contagem
    frame_vis = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.putText(frame_vis, f"Capturas: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Stream Hikvision - 's' para capturar, 'q' para sair", frame_vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Captura finalizada pelo usuário.")
        break
    elif key == ord('s'):
        try:
            response = requests.get(url_snapshot, auth=HTTPDigestAuth(username, password), stream=True)
        except Exception as e:
            print(f"Erro ao obter snapshot: {e}")
            break

        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            filename = os.path.join(save_dir, f"imagem_{count:03d}.jpg")
            cv2.imwrite(filename, img)

            # Envia comando ao Arduino
            ser.write(b'R')
            ser.flush()
            print(f"[✓] Imagem salva: {filename}")
            print("[→] Comando 'R' enviado para girar o motor.")
            count += 1
        else:
            print(f"Erro ao capturar imagem da câmera. Código HTTP: {response.status_code}")

cap.release()
cv2.destroyAllWindows()
ser.close()
