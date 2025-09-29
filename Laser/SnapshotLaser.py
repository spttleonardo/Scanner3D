import os
import time
from requests.auth import HTTPDigestAuth
import cv2
import numpy as np
import requests
import configparser
# import structuredlight as sl
import time

# ----------------- CONFIG ZONE -------------------------------------------------------------------
camera_ip = '192.168.1.64'
username = 'admin'
password = 'egl@1234'
porta_rtsp = "554"

url = f'http://{camera_ip}/ISAPI/Streaming/channels/101/picture'
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{porta_rtsp}/Streaming/Channels/101"

fps = 10  # quantas fotos por segundo (cuidado: não exagere, depende da câmera)
duration = 5  # segundos de vídeo
num_frames = fps * duration

output_path = './assets/video_snapshot.avi'
base_path = './assets/laserPlan'

numPhoto = 4 # number of calibration photos + 1
setTime = 6   # number of seconds between snapshots + 1

# ------------- Function to find the next available folder name ----------------
def get_next_folder(base_path, base_name='object_'):
    i = 0
    while True:
        folder_name = f"{base_name}{chr(97+i)}"  # 97 = 'a'
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)  # Creates new folder
            return full_path
        i += 1

# ------------- END OF CONFIG ZONE ----------------------------------------------------------------

choice = input("Escolha o tipo de snapshot" +
               f"\n[1]: Gravação de vídeo de {duration} segundos a com {fps} fps)" +
               "\n[2]: Cena (uma imagem da cena dentro do BB) " + 
               f"\n[3]: Tirar {numPhoto-1} fotos da cena com o Laser em intervalos de {setTime-1} segundos" +
               f"\n[4]: Tirar {numPhoto-1} fotos da cena com o Laser através do Stream aberto" +
               f"\n[5]: Tirar 3 fotos da cena automaticamente com Stripes" +
               "\n Sua escolha: ")

# -------------------------------------------------------------------------------------------------
if choice == '1':

    # Verifica pasta
    # os.makedirs('./assets', exist_ok=True)

    # Primeiro frame para pegar tamanho
    response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
    if response.status_code != 200:
        print("Erro ao capturar primeiro frame.")
        exit()

    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    height, width, _ = frame.shape

    # Inicializa o writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Gravando vídeo de snapshots...")

    for i in range(num_frames):
        response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            out.write(frame)
            print(f"Frame {i+1}/{num_frames} capturado.")
        else:
            print(f"Falha no frame {i+1}")
        
        time.sleep(1 / fps)  # espera para manter o fps

    out.release()
    print(f"Vídeo salvo em {output_path}")

# -------------------------------------------------------------------------------------------------
elif choice == '2':
    
    # Open stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Erro ao conectar ao stream RTSP.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame.")
            break

        # Resize frame
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # Size of bounding box
        box_width = 600
        box_height = 300

        # Size of image
        h, w, _ = frame.shape

        # Coordinates of bounding box centralized
        x1 = (w - box_width) // 2
        y1 = (h - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Draw bounding box (Green line, Thickness 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Pressione 's' para snapshot e 'q' para sair", frame)

        # Press 's' to take a snapshot or 'q' to leave
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                # Try to take snapshot
                response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
                print("response:", response)
                print(f"Imagem capturada com sucesso!")
            except:
                print(f"Erro ao conectar à câmera: {Exception.with_traceback()} | {Exception.__traceback__}")
                exit()

            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Save image as simple RAW (Data pure RGB)
                raw_data = img.tobytes()
                with open('./assets/cena/cena1.raw', 'wb') as f:
                    f.write(raw_data)
                
                # Save as JPG image
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

    # Creates the next folder available to save snapshots
    save_folder = get_next_folder(base_path)

    for i in range(1, numPhoto):

        for j in range(1, setTime):
            # Counter untill next snapshot
            print(f"Contagem até próximo snapshot: {j}")
            cv2.waitKey(1000)

        try:
            # Try to take snapshot
            response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
            print("response:", response)
            print(f"Imagem {i} capturada com sucesso!")
        except:
            print(f"Erro ao conectar à câmera: {Exception.with_traceback()} | {Exception.__traceback__}")
            exit()

        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Save image as simple RAW (Data pure RGB)
            raw_data = img.tobytes()
            with open(f'./assets/obj{chr(96+i)}/snapshot.raw', 'wb') as f:
                f.write(raw_data)
            
            # Save as JPG image
            cv2.imwrite(f'{save_folder}/snapshot_{chr(96+i)}.jpg', img)

            print("Snapshot salvo como .raw com sucesso!")
        else:
            print(f"Erro ao capturar snapshot: {response.status_code}")


# -------------------------------------------------------------------------------------------------
elif choice == '4':

    # Open the stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error connecting to stream RTSP.")
        exit()

    # Create the folder to save the snapshots
    # save_folder = get_next_folder(base_path)

    i = 1  # Counter for the number of photos takens

    while i < numPhoto:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o frame.")
            break

        # Resize frame
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # Draw bounding box (Green line, Thickness 2)
        box_width = 600
        box_height = 300
        h, w, _ = frame.shape
        x1 = (w - box_width) // 2
        y1 = (h - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Pressione 's' para tirar foto, 'q' para sair", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            try:
                # Take snapshot
                response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
                print(f"Tentando capturar imagem {i}...")
            except Exception as e:
                print(f"Erro ao conectar à câmera: {e}")
                exit()

            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Save image as simple RAW (Data pure RGB)
                # raw_data = img.tobytes()
                # with open(f'{save_folder}/snapshot_{chr(96+i)}.raw', 'wb') as f:
                #     f.write(raw_data)

                # Save as JPG image
                cv2.imwrite(f'{base_path}/snapshot_{chr(96+i)}.jpg', img)

                print(f"Snapshot {i} salvo com sucesso!")
                i += 1  # Increments the photo counter
            else:
                print(f"Erro ao capturar snapshot: {response.status_code}")

        elif key == ord('q'):
            print("Saindo...")
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------
elif choice == '5':

    # Creates the next folder available to save snapshots
    save_folder = get_next_folder(base_path)

    print(f"Salvando imagens em {save_folder}...")

    # Inicializa câmera
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     raise IOError("Não foi possível acessar a câmera.")

    # Define resolução dos padrões
    width, height = 640, 480

    # Gera padrões de luz estruturada
    # stripe = sl.Stripe()
    # imlist = stripe.generate((width, height))

    # binary = sl.Binary()
    # imlist = binary.generate((width, height))
    # img_index = binary.decode(imlist, thresh=127)

    gray = 0 #sl.Gray()
    imlist = gray.generate((width, height))
    img_index = gray.decode(imlist, thresh=127)


    # Prepara janela em tela cheia
    cv2.namedWindow("FullScreen", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("FullScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    captured_images = []

    for i, img in enumerate(imlist):
        cv2.imshow("FullScreen", img)
        cv2.waitKey(0)
        # cv.waitKey(1000)   # aguarda 500ms (ajuste conforme necessário)

        # ret, frame = cap.read()
        # if ret:
        #     captured_images.append(frame)
        #     # cv.imwrite(f"captura_{i:02d}.png", frame)
        # else:
        #     print(f"Erro ao capturar frame {i}")

        # Snapshots Zone -------------------------------------------------------------------------------

        # for j in range(1, setTime):
        #     # Counter untill next snapshot
        #     print(f"Contagem até próximo snapshot: {j}")
        #     cv2.waitKey(1000)

        try:
            # Try to take snapshot
            response = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
            print("response:", response)
            print(f"Imagem {i} capturada com sucesso!")
        except:
            print(f"Erro ao conectar à câmera: {Exception.with_traceback()} | {Exception.__traceback__}")
            exit()

        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Save image as simple RAW (Data pure RGB)
            # raw_data = img.tobytes()
            # with open(f'{save_folder}/snapshot{chr(96+i)}.raw', 'wb') as f:
            #     f.write(raw_data)
            
            # Save as JPG image
            cv2.imwrite(f'{save_folder}/snapshot_{chr(97+i)}.jpg', img)

            print("Snapshot salvo como .raw com sucesso!")
        else:
            print(f"Erro ao capturar snapshot: {response.status_code}")
            break

    # cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------

else:
    print("Opção inválida. Saindo...")
