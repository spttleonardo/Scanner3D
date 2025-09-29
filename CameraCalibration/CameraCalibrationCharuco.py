import cv2
import numpy as np
import glob
import os
import sys

# ————————————— USER CONFIG —————————————
IMAGE_DIR     = './assets/cena/'   # pasta com as fotos do Charuco board
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
SQUARES_X     = 7                  # quadrados internos (cols)
SQUARES_Y     = 7                  # quadrados internos (rows)
SQUARE_LEN    = 0.027           # tamanho do quadrado, em metros
MARKER_LEN    = 0.0168             # tamanho do marker, em metros
OUTPUT_FILE   = 'camera_calibration_charuco.npz'
# —————————————————————————————————————————————————

def main():
    # 1) cria CharucoBoard & detector
    aruco_dict  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board       = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                         SQUARE_LEN, MARKER_LEN,
                                         aruco_dict)
    det_params  = cv2.aruco.DetectorParameters()
    
    all_corners = []
    all_ids     = []
    img_size    = None
    
    # 2) detecta em cada imagem
    for img_path in glob.glob(os.path.join(IMAGE_DIR, '*.*')):
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Não abriu {img_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]
        
        # detecta markers e interpola charuco
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=det_params)
        if ids is None or len(ids) < 4:
            print(f"[!] <4 marcadores em {os.path.basename(img_path)}, pulando")
            continue
        
        ret, charu_corners, charu_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        
        if charu_corners is None or len(charu_corners) < 4:
            print(f"[!] <4 cantos Charuco em {os.path.basename(img_path)}, pulando")
            continue
        
        all_corners.append(charu_corners)
        
        all_ids.append(charu_ids)
        

        # visualização
        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, charu_corners, charu_ids)
        disp = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Detect & Interpolate', disp)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    if not all_corners:
        print("❌ Nenhum canto Charuco detectado. Cheque as imagens e o board.")
        sys.exit(1)
    
    # 3) calibração
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(rvecs)
    
    # 4) resultados
    print(f"\n✅ RMS reproj error: {ret:.4f} px")
    print("📷 Intrinsic matrix (K):\n", K)
    print("🎯 Distortion coeffs:\n", dist.ravel())
    
    # 5) opcional: teste de undistort e exibição de retas
    test_undistort(K, dist, img_size)
    
    # 6) salva parâmetros
    np.savez(OUTPUT_FILE,
             mtx=K, dist=dist,
             rvecs=rvecs, tvecs=tvecs,
             reprojError=ret)
    print(f"\n🗄️  Parâmetros salvos em '{OUTPUT_FILE}'")

def test_undistort(K, dist, img_size):
    # gera imagem de teste vazia com linhas retas
    w, h = img_size
    # cria grelha de retas horizontais e verticais
    test = np.zeros((h, w, 3), dtype=np.uint8)
    spacing = 50
    for x in range(0, w, spacing):
        cv2.line(test, (x,0), (x,h), (255,255,255), 1)
    for y in range(0, h, spacing):

        cv2.line(test, (0,y), (w,y), (255,255,255), 1)
    # undistort
    und = cv2.undistort(test, K, dist, None)
    test = cv2.resize(test, dsize=None, fx=0.5, fy=0.5)
    und = cv2.resize(und, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow('Original grid', test)
    cv2.imshow('Undistorted grid', und)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
