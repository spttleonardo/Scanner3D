import cv2
import numpy as np

# ————————————————————————— CONFIGURAÇÃO —————————————————————————
# Três imagens do CharUco interceptado pelo laser, em poses A, B e C:
IMG_A    = './assets/laserPlan/snapshot_a.jpg' # otima imagem parabens
IMG_B    = './assets/laserPlan/snapshot_b.jpg'
IMG_C    = './assets/laserPlan/snapshot_c.jpg'
# (Opcional) imagem para validar o plano depois:
IMG_TEST = './assets/laserPlan/snapshot_c.jpg'

# Parâmetros do CharUco board (idem aos usados na calibração)
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
SQUARES_X     = 7
SQUARES_Y     = 7
SQUARE_LEN    = 0.027   # em metros
MARKER_LEN    = 0.0168  # em metros

# Arquivo com a calibração prévia
CALIB_FILE = './CameraCalibration/camera_calibration_charuco.npz'

# ————————————————————————————————————————————————————————————————

# 1) Carrega calibração
data = np.load(CALIB_FILE)
mtx, dist = data['mtx'], data['dist']

# 2) Prepara CharUco e DetectorParameters
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board      = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                    SQUARE_LEN, MARKER_LEN,
                                    aruco_dict)
det_params = cv2.aruco.DetectorParameters()

# ————————————————————————— FUNÇÕES —————————————————————————

def estimate_board_pose(img_path):
    """
    Lê imagem, tira distorção, detecta CharUco e retorna:
    undistorted color, gray, rvec, tvec, e charu_corners.
    """
    img = cv2.imread(img_path)
    und = cv2.undistort(img, mtx, dist, None)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict,
                                              parameters=det_params)
    _, charu_corners, charu_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board
    )
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
        raise RuntimeError(f"Pose não encontrada em {img_path}")

    # Desenho rápido para validar
    vis = und.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.aruco.drawDetectedCornersCharuco(vis, charu_corners, charu_ids)
    cv2.drawFrameAxes(vis, mtx, dist, rvec, tvec, 0.1)
    disp = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Pose CharUco", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return und, gray, rvec, tvec, charu_corners

def detectar_linha_laser(im_bgr, limiar=200):
    """
    Segmentar o laser (vermelho) encontrando em cada linha y
    o pixel x de maior diferença R - (G+B)/2 > limiar.
    """
    b, g, r = cv2.split(im_bgr)
    idiff = r.astype(np.float32) - ((g.astype(np.float32) + b.astype(np.float32)) / 2)
    #cv2.imshow('idiff', idiff)
    # r = cv2.GaussianBlur(r, (5,1), 0)

    h, w = r.shape
    mascara = (r > limiar).astype(np.uint8) * 255
    mask = cv2.resize(mascara, None, fx=0.5, fy=0.5)
    cv2.imshow('Mascara R > limiar', mask)

    # h, w = idiff.shape
    pts = []
    for y in range(h):
        linha = mascara[y]
        x = int(np.argmax(linha))
        if linha[x] > limiar or linha[x] +10 > limiar:
            pts.append((x,y))
    return pts


def pixel_to_cam(pt_uv, rvec, tvec):
    """
    Converte pixel (u,v) na imagem corrigida para um ponto Xc em 3D
    no referencial da CÂMERA, interceptando o plano do board.
    """
    # 1) coordenadas normalizadas do pixel
    uvn = cv2.undistortPoints(
        np.array([[pt_uv]], np.float32),
        mtx, dist
    )[0,0]  # (x_norm, y_norm)

    # 2) definiu o ray na câmera
    ray = np.array([uvn[0], uvn[1], 1.0])
  

    # 3) normal do plano do board em coords câmera = 3ª coluna de R
    R, _ = cv2.Rodrigues(rvec)
    n_b = R[:,2]

    # 4) fator de escala para que n_b·Xc = n_b·tvec
    rhs   = n_b.dot(tvec.flatten())
    denom = n_b.dot(ray)
    s     = rhs / denom

    # 5) ponto em 3D na câmera
    Xc = s * ray
    return Xc

def detectar_linha_laser_subtracao(im_bgr, background_bgr, limiar=30):
    diff = cv2.absdiff(im_bgr, background_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, limiar, 255, cv2.THRESH_BINARY)
    h, w = mask.shape
    pts = []
    for y in range(h):
        xs = np.where(mask[y]>0)[0]
        if len(xs):
            x = int(np.mean(xs))
            pts.append((x,y))
    cv2.imshow("Mask subtração", mask)
    return pts


def get_two_3d_points(und, gray, rvec, tvec, charu_corners):
    """
    Extrai dois pontos 3D da reta do laser, filtrando só os pixels
    que caem dentro do polígono dado pelos charu_corners.
    """
    pts2d = detectar_linha_laser(und)
    poly  = charu_corners.reshape(-1,2).astype(int)

    # filtra dentro do board
    pts_f = [pt for pt in pts2d
             if cv2.pointPolygonTest(poly, pt, False) >= 0]
    if len(pts_f) < 2:
        raise RuntimeError("Poucos pontos do laser dentro do board")

    # escolhe Y min/máx para extremos
    ys     = [y for _,y in pts_f]
    y_min, y_max = min(ys), max(ys)
    x_top = [x for x,y in pts_f if y==y_min][0]
    x_bot = [x for x,y in pts_f if y==y_max][0]

    # converte
    P1 = pixel_to_cam((x_top, y_min), rvec, tvec)
    P2 = pixel_to_cam((x_bot, y_max), rvec, tvec)
    return P1, P2

# ————————————————————————— PROGRAMA PRINCIPAL —————————————————————————

# 3) Processa A, B e C
pcs = []
for path in (IMG_A, IMG_B, IMG_C):
    und, gray, rvec, tvec, charu_corners = estimate_board_pose(path)
    P1, P2 = get_two_3d_points(und, gray, rvec, tvec, charu_corners)
    pcs.append((P1, P2))
    print(f"{path}\n P1 = {P1}\n P2 = {P2}\n")

# 4) Ajusta o plano do laser no referencial da câmera
PA1, _ = pcs[0]
PB1, _ = pcs[1]
PC1, _ = pcs[2]

v1     = PB1 - PA1
v2     = PC1 - PA1
normal = np.cross(v1, v2)
normal /= np.linalg.norm(normal)
d      = -normal.dot(PA1)

print("Plano do laser em coords da câmera:")
print("  normal =", normal)
print("  d      =", d)

# 5) Validação opcional numa 4ª imagem
und_t, gray_t, rvec_t, tvec_t, _ = estimate_board_pose(IMG_TEST)
pts2  = detectar_linha_laser(und_t)
vis    = und_t.copy()
for (u,v) in pts2:
    Xc = pixel_to_cam((u,v), rvec_t, tvec_t)
    dist_plane = normal.dot(Xc) + d
    color = (0,255,0) if abs(dist_plane) < 1e-3 else (0,0,255)
    cv2.circle(vis, (u,v), 2, color, -1)


cv2.imshow("Validação Plano Laser", cv2.resize(vis,(0,0),fx=0.5,fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
