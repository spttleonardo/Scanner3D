import cv2
import numpy as np
import glob
# import visaoComputacional as visco

# Parâmetros do tabuleiro de xadrez
chessboard_size = (7,7)  # número de cantos internos (colunas, linhas)
square_size = 2.5     # tamanho do quadrado (em cm, mm, ou o que você escolher)

# Critério de precisão para cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vetores para armazenar os pontos 3D do mundo real e os pontos 2D da imagem
objpoints = []  # pontos 3D
imgpoints = []  # pontos 2D
image_name = []  # nomes das imagens

# Preparar os pontos do tabuleiro, por exemplo (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)*square_size
# objp *= square_size

# Carregar imagens do tabuleiro
images = glob.glob('assets/*.jpg')  # coloque suas imagens na pasta correta
print("imagens encontradas: ", len(images))

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        imgpoints.append(corners2)
        image_name.append(fname.split('/')[-1]) # Extrai o nome do arquivo da imagem

        # Desenhar e mostrar os cantos
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(0)
    else:
        print(f"Cantos não encontrados para {fname}")

# cv2.destroyAllWindows()

# Calibrar a câmera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Salvar resultados
np.savez("camera_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, imgName=image_name)

print("Matriz da câmera:\n", mtx)
print("Coeficientes de distorção:\n", dist)

# Carrega o arquivo
data = np.load("camera_calibration.npz")

# Lista os nomes das variáveis salvas
print("---------------------------------")
print("\nChaves disponíveis no arquivo:")
print(data.files)

# Exibe o conteúdo de cada uma
for key in data.files:
    print(f"\nConteúdo de '{key}':")
    print(data[key])

# ---------------------------------------------------------
