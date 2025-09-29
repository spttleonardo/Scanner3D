from requests.auth import HTTPDigestAuth
import cv2
import numpy as np
import requests

# ----------------- CONFIG ZONE -------------------------------------------------------------------
camera_ip = '10.1.1.72'
username = 'admin'
password = '123456789A'
porta_rtsp = "554"

# URL de snapshot para câmeras Intelbras
url = f'http://{camera_ip}/cgi-bin/snapshot.cgi'
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{porta_rtsp}/Streaming/Channels/101"

numPhoto = 14  # número de fotos de calibração + 1
setTime = 6    # segundos entre fotos + 1
# ------------- END OF CONFIG ZONE ----------------------------------------------------------------

choice = input("Escolha o tipo de snapshot" + 
               f"\n[1]: Calibração ({numPhoto-1} fotos em intervalos de {setTime-1} segundos)" +
               "\n[2]: Cena (uma imagem da cena dentro do BB) " + 
               "\n Sua escolha: ")

if choice == '1':
    for i in range(1, numPhoto):
        try:
            response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
            print(f"Imagem {i} capturada com sucesso!")
        except Exception as e:
            print(f"Erro ao conectar à câmera: {e}")
            exit()

        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            raw_data = img.tobytes()
            with open('./assets/snapshotintel.raw', 'wb') as f:
                f.write(raw_data)

            cv2.imwrite(f'./assets/snapshotintel_{chr(96+i)}.jpg', img)
            print("Snapshot salvo como .raw com sucesso!")
        else:
            print(f"Erro ao capturar snapshot: {response.status_code}")

        for j in range(1, setTime):
            print(f"Contagem até próximo snapshot: {j}")
            cv2.waitKey(1000)

elif choice == '2':
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Erro ao conectar ao stream RTSP.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame.")
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        box_width = 600
        box_height = 300
        h, w, _ = frame.shape
        x1 = (w - box_width) // 2
        y1 = (h - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Pressione 's' para snapshot e 'q' para sair", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
                print("Imagem capturada com sucesso!")
            except Exception as e:
                print(f"Erro ao conectar à câmera: {e}")
                exit()

            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                raw_data = img.tobytes()
                with open('./assets/cena/cena1.raw', 'wb') as f:
                    f.write(raw_data)

                cv2.imwrite(f'./assets/cena/cena1.jpg', img)
                print("Snapshot salvo como .raw com sucesso!")
            else:
                print(f"Erro ao capturar snapshot: {response.status_code}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
