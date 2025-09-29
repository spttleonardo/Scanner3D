import cv2
import numpy as np

# ————————————— Ajuste só isto —————————————
IMG_BOARD      = './assets/cena/cena1.jpg'  # seu tabuleiro impresso e fotografado
CALIB_FILE     = './CameraCalibration/camera_calibration_charuco.npz'
ARUCO_DICT_ID  = cv2.aruco.DICT_6X6_250
SQUARES_X      = 7       # igual ao que você definiu
SQUARES_Y      = 7
SQUARE_LENGTH  = 0.027   # em metros (3 cm)
MARKER_LENGTH  = 0.0168  # em metros
AXIS_LENGTH    = 0.05    # tamanho dos eixos desenhados (m)
# —————————————————————————————————————————————————

# 1) carrega calibração
with np.load(CALIB_FILE) as data:
    mtx, dist = data['mtx'], data['dist']

# 2) prepara CharucoBoard e detector
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board      = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                    SQUARE_LENGTH, MARKER_LENGTH,
                                    aruco_dict)
det_params = cv2.aruco.DetectorParameters()

# 3) lê imagem do tabuleiro (fotografia)
img = cv2.imread(IMG_BOARD)
if img is None:
    raise FileNotFoundError(f"Não achei '{IMG_BOARD}'")
und = cv2.undistort(img, mtx, dist, None)
gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

# 4) detecta marcadores ArUco e refina cantos ChArUco
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=det_params)
if ids is None or len(ids) == 0:
    raise RuntimeError("Nenhum marcador ArUco detectado!")
ret, charu_corners, charu_ids = cv2.aruco.interpolateCornersCharuco(
    markerCorners=corners,
    markerIds=ids,
    image=gray,
    board=board
)
if not ret or len(charu_ids) < 4:
    raise RuntimeError("Não há cantos ChArUco suficientes para estimar pose!")

# 5) estima pose do CharucoBoard
ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
    charucoCorners=charu_corners,
    charucoIds=charu_ids,
    board=board,
    cameraMatrix=mtx,
    distCoeffs=dist,
    rvec=None,
    tvec=None
)
if not ok:
    raise RuntimeError("Falha ao estimar pose do CharUcoBoard.")

# 6) desenha tudo
vis = und.copy()
cv2.aruco.drawDetectedMarkers(vis, corners, ids)
cv2.aruco.drawDetectedCornersCharuco(vis, charu_corners, charu_ids)
cv2.drawFrameAxes(vis, mtx, dist, rvec, tvec, AXIS_LENGTH)

# 7) exibe resultado reduzido
disp = cv2.resize(vis, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
cv2.imshow("CharucoBoard + Pose", disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8) imprime rvec / tvec
print("→ Vetor de rotação (rvec):", rvec.ravel())
print("→ Vetor de translação (tvec):", tvec.ravel())
