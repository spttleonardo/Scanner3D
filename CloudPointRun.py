import os
import math
import cv2
import numpy as np
import open3d as o3d

# ————— Parâmetros gerais —————
CALIB_FILE   = './CameraCalibration/camera_calibration_charuco.npz'

# Configuração videos 1 até 3
REF_RVEC    = np.array([-0.04539768,  0.49278872, -1.55983302])
REF_TVEC    = np.array([-0.0058707,   0.10990249,  0.47977395])
LASER_NORM  = np.array([0.97771972, 0.09235763,  -0.18850524])
LASER_D     = 0.09127957687542472
Y_MIN, Y_MAX = 300, 900
X_MIN, X_MAX = 300, 1000

# Configuração videos 5 até 23
# REF_TVEC    = np.array([0.01266874, 0.10931843, 0.49754032])
# REF_RVEC    = np.array([-0.05082698,  0.46228681, -1.52721153])
# LASER_NORM  = np.array([0.96550531,  0.0779552,  -0.24844009])
# LASER_D     = 0.10492318481042316
# Y_MIN, Y_MAX = 0, 900
# X_MIN, X_MAX = 300, 1000


# Configuração videos 24 até 35
# REF_RVEC    = np.array([-0.12992581, 0.40412338, -1.52086446])
# REF_TVEC    = np.array([0.01266874, 0.10931843, 0.49754032])
# LASER_NORM  = np.array([0.96742069, 0.06058479, -0.24581843])
# LASER_D     = 0.11423071255976615
# Y_MIN, Y_MAX = 300, 900
# X_MIN, X_MAX = 300, 1000


VIDEO_PATH   = './assets/videos/dataset/video_001.avi'
THRESH       = 170
U0    = 913
DELTA = 50

# —————————————————————————————————————————————

# 1) Carrega calibração e monta R_ref
data = np.load(CALIB_FILE)
mtx, dist = data['mtx'], data['dist']
R_ref, _  = cv2.Rodrigues(REF_RVEC)

def scan_profile(frame):


    # Extrai somente o canal vermelho e exibe em escala de cinza
    # r_channel = frame[:, :, 2]
    # cv2.imshow("Canal R (vermelho) - Grayscale", r_channel)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Canal R (vermelho) - Grayscale")


    und = cv2.undistort(frame, mtx, dist, None)
    _, _, r = cv2.split(und)
    _, mask = cv2.threshold(r, THRESH, 255, cv2.THRESH_BINARY)

    # kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,8))
    # kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,8))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 12))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)

    roi_rect = np.zeros_like(mask)
    roi_rect[Y_MIN:Y_MAX, X_MIN:X_MAX] = 255
    mask = cv2.bitwise_and(mask, roi_rect)

    h, w = mask.shape
    u_min = max(0, U0 - DELTA)
    u_max = min(w, U0 + DELTA)
    roi_vert = np.zeros_like(mask)
    roi_vert[:, u_min:u_max] = 255
    mask = cv2.bitwise_and(mask, roi_vert)

    pts3d = []
    for v in range(mask.shape[0]):
        x_indices = np.where(mask[v, :] == 255)[0]
        if len(x_indices) > 0:
            u = int(np.mean(x_indices))
            uvn = cv2.undistortPoints(
                np.array([[[u, v]]], np.float32),
                mtx, dist
            )[0,0]
            ray = np.array([uvn[0], uvn[1], 1.0], dtype=float)
            t = -LASER_D / float(LASER_NORM.dot(ray))
            pts3d.append(ray * t)

    if len(pts3d) == 0:
        return np.empty((0,3), dtype=float)
    return np.vstack(pts3d)

# 2) Abre vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Erro ao abrir vídeo: {VIDEO_PATH}")

N_SCANS = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ANGLE_STEP = 360.0 / N_SCANS
print(f"[✓] Frames detectados: {N_SCANS}, Ângulo por frame: {ANGLE_STEP:.2f}°")

all_slices = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    prof_cam = scan_profile(frame)
    if prof_cam.shape[0] == 0:
        print(f"Frame {frame_idx:03d}: nenhum ponto → pulando")
        frame_idx += 1
        continue

    prof_board = (R_ref.T @ (prof_cam - REF_TVEC).T).T

    ang = ANGLE_STEP * frame_idx
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    R_x = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=float)

    prof_final = prof_board @ R_x.T
    all_slices.append(prof_final)

    print(f"Frame {frame_idx:03d}: {prof_cam.shape[0]:5d} pts, X rot={ang:.1f}°")
    frame_idx += 1

cap.release()

# 3) Concatena e exibe
if not all_slices:
    raise RuntimeError("Nenhum ponto reconstruído.")

cloud = np.vstack(all_slices)
print(f"Total de pontos na nuvem: {cloud.shape}")

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
pcd.paint_uniform_color([1,0,0])
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

o3d.visualization.draw_geometries(
    [pcd, axis],
    window_name='Reconstrução 3D — ROI + Rot X',
    width=800, height=600
)

o3d.io.write_point_cloud("output_cloud.ply", pcd)
print("Nuvem de pontos salva em output_cloud.ply")

cv2.destroyAllWindows()
