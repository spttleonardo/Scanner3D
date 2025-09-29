from requests.auth import HTTPDigestAuth
import cv2
import numpy as np
import requests
import configparser

# ----------------- CONFIG ZONE -------------------------------------------------------------------
camera_ip = '192.168.1.64'
username = 'admin'
password = 'egl@1234'
porta_rtsp = "554"

url = f'http://{camera_ip}/ISAPI/Streaming/channels/101/picture'
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{porta_rtsp}/Streaming/Channels/101"

numPhoto = 14 # number of calibration photos + 1
setTime = 6   # number of seconds between snapshots + 1
# ------------- END OF CONFIG ZONE ----------------------------------------------------------------

choice = input("Escolha o tipo de snapshot" + 
               f"\n[1]: Calibração ({numPhoto-1} fotos em intervalos de {setTime-1} segundos)" +
               "\n[2]: Cena (uma imagem da cena dentro do BB) " + 
               "\n[3]: Abre stream RTSP e captura imagens pressionando 's'" +
               "\n Sua escolha: ")

if choice == '1':

    for i in range(1, numPhoto):

        try:
            # Tente obter o snapshot da câmera
            response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
            print("response:", response)
            print(f"Imagem {i} capturada com sucesso!")
        except:
            print(f"Erro ao conectar à câmera: {Exception.with_traceback()} | {Exception.__traceback__}")
            exit()

        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Salvar como imagem RAW simples (dados RGB puros)
            raw_data = img.tobytes()
            with open('./assets/snapshot.raw', 'wb') as f:
                f.write(raw_data)
            
            # Salvar como imagem JPG
            cv2.imwrite(f'./assets/snapshot_{chr(96+i)}.jpg', img)

            print("Snapshot salvo como .raw com sucesso!")
        else:
            print(f"Erro ao capturar snapshot: {response.status_code}")

        for j in range(1, setTime):
            # Contador até próximo snapshot
            print(f"Contagem até próximo snapshot: {j}")
            cv2.waitKey(1000)

# -------------------------------------------------------------------------------------------------
elif choice == '2':
    
    # Abre o stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Erro ao conectar ao stream RTSP.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame.")
            break

        # Redimensionar o frame
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # Tamanho do bounding box
        box_width = 600
        box_height = 300

        # Tamanho da imagem
        h, w, _ = frame.shape

        # Coordenadas do bounding box centralizado
        x1 = (w - box_width) // 2
        y1 = (h - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Desenha o bounding box (linha verde, espessura 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Pressione 's' para snapshot e 'q' para sair", frame)

        # Pressione 's' para tirar snapshot ou 'q' para sair
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                # Tente obter o snapshot da câmera
                response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
                print("response:", response)
                print(f"Imagem capturada com sucesso!")
            except:
                print(f"Erro ao conectar à câmera: {Exception.with_traceback()} | {Exception.__traceback__}")
                exit()

            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Salvar como imagem RAW simples (dados RGB puros)
                raw_data = img.tobytes()
                with open('./assets/cena/cena1.raw', 'wb') as f:
                    f.write(raw_data)
                
                # Salvar como imagem JPG
                cv2.imwrite(f'./assets/cena/cena1.jpg', img)

                print("Snapshot salvo como .raw com sucesso!")
            else:
                print(f"Erro ao capturar snapshot: {response.status_code}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------
elif choice == '3':
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Erro ao conectar ao stream RTSP.")
        exit()

    i = 1  # Contador de imagens
    print("Modo interativo iniciado: pressione 's' para capturar e 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame.")
            break

        # Redimensiona frame para visualização
        frame_vis = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Mostra número de capturas
        cv2.putText(frame_vis, f"Capturas: {i - 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pressione 's' para capturar | 'q' para sair", frame_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Captura finalizada pelo usuário.")
            break
        elif key == ord('s'):
            try:
                response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
            except Exception as e:
                print(f"Erro ao conectar à câmera: {e}")
                break

            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                letra = chr(96 + i) if i <= 26 else str(i)  # 'a'-'z', depois '27', '28',...
                cv2.imwrite(f'./assets/snapshot_{letra}.jpg', img)

                raw_data = img.tobytes()
                with open('./assets/snapshot.raw', 'wb') as f:
                    f.write(raw_data)

                print(f"Snapshot {letra} salvo com sucesso!")
                i += 1
            else:
                print(f"Erro ao capturar snapshot: {response.status_code}")

    cap.release()
    cv2.destroyAllWindows()
