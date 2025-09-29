import cv2

video_path = './assets/videos/video_006.avi'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[!] Falha ao abrir o vídeo: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[✓] Fim do vídeo.")
        break

    # Redimensiona o frame para 1/5 do tamanho original
    frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Visualizador FFV1 (1/2)", frame_resized)
    
    if cv2.waitKey(30) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
