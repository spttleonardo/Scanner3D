import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import serial
import os
import time

# ---------------- CONFIG ----------------
camera_ip = '192.168.1.64'
username = 'admin'
password = 'egl@1234'
snapshot_url = f"http://{camera_ip}/ISAPI/Streaming/channels/101/picture"

PORT = 'COM4'
BAUD = 9600

output_dir = './assets/videos/dataset'
os.makedirs(output_dir, exist_ok=True)

# Função para gerar o próximo nome de vídeo
def get_next_video_name(output_dir, base_name='video_', ext='.avi'):
    i = 1
    while True:
        name = f"{base_name}{i:03d}{ext}"
        full_path = os.path.join(output_dir, name)
        if not os.path.exists(full_path):
            return full_path
        i += 1

# ---------------- INICIALIZAÇÕES ----------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("[✓] Conectado ao Arduino.")

video_name = get_next_video_name(output_dir)
print(f"[→] Próximo vídeo será salvo como: {video_name}")

out = None
fps = 10  # ajuste conforme desejado

# ---------------- LOOP PRINCIPAL ----------------
while True:
    # Tira snapshot
    try:
        response = requests.get(snapshot_url, auth=HTTPDigestAuth(username, password), stream=True, timeout=2)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("[!] Falha ao decodificar imagem.")
                continue

            h, w = frame.shape[:2]

            if out is None:
                # Inicia o vídeo writer
                out = cv2.VideoWriter(
                    video_name,
                    cv2.VideoWriter_fourcc(*'FFV1'),  # MJPEG FFV1
                    fps,
                    (w, h)
                )
                if not out.isOpened():
                    print("[!] Falha ao criar o vídeo.")
                    break

            out.write(frame)
            print(f"[✓] Frame adicionado ao vídeo.")

            # Manda comando para o Arduino
            ser.write(b'R')
            ser.flush()
            print("[→] Comando R enviado ao Arduino.")

        else:
            print(f"[!] HTTP {response.status_code} na captura do snapshot.")

    except Exception as e:
        print(f"[!] Erro no snapshot: {e}")
        continue

    # Espera resposta do Arduino
    while True:
        if ser.in_waiting:
            msg = ser.readline().decode().strip()
            print(f"[Arduino] {msg}")
            if msg == 'C':
                # Pode capturar o próximo
                break
            elif msg == 'Q':
                # Finaliza vídeo
                print("[✓] Sinal de 360º recebido. Finalizando vídeo.")
                out.release()
                ser.close()
                print(f"[✓] Vídeo salvo em: {video_name}")
                exit()

