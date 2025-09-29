import cv2
import numpy as np
import glob
import visaoComputacional as visco

# Abrir o arquivo .npz salvo anteriormente
# with np.load("camera_calibration.npz") as data:
with np.load("camera_calibration_charuco.npz") as data:
    mtx = data['mtx']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']

print("Dados carregados do arquivo .npz:")
print("Matriz da câmera:\n", mtx)
print("Coeficientes de distorção:\n", dist)

indice_imagem = 0 #4

# Matriz de rotacao
vetor_rotacao = rvecs[indice_imagem]
R, _ = cv2.Rodrigues(vetor_rotacao)
print('Matriz de rotacao:\n', R)

# print('\nrvecs: ', rvecs)

# vetor de translacao
vetor_translacao = tvecs[indice_imagem]
print('Vetor de translacao:\n', vetor_translacao)

# obtendo a matriz de projecao
K = np.zeros((3,4), np.float32)
K[0:3, 0:3] = mtx

M = np.zeros((4,4), np.float32)
M[0:3, 0:3] = R
M[0:3,3] = -vetor_translacao[:,0]
M[3,3] = 1

# Matriz de projecao
P = np.linalg.matmul(K,M)

print('Matriz de projecao\n', P)

P2 = np.delete(P,2,1)

print('Matriz de projecao reduzida\n', P2)

# ---------------------------------------------

I2 = cv2.imread('assets\cena\cena1.jpg')
I2 = cv2.resize(I2, (0,0), fx=0.5, fy=0.5)

cv2.imshow('Imagem reduzida', I2)

# corrigindo distorções da lente
I3 = cv2.undistort(I2, mtx, dist, None)
cv2.imshow('Imagem \'sem\' distorcoes', I3)

I4 = cv2.cvtColor(I3, cv2.COLOR_BGR2GRAY)

ret, I5 = cv2.threshold(I4, 50, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Imagem limiarizada', I5)

I6 = visco.imclearboard(I5, 40)

marker = I6.copy()
# marker[1+margem:-1-300, 1+250:-1-200] = 0 # (y,x)
# Tamanho do bounding box
box_width = 500 #200
box_height = 300 #150

# Tamanho da imagem
h, w, _ = I2.shape

# Coordenadas do bounding box centralizado
x1 = (w - box_width) // 2
y1 = (h - box_height) // 2
x2 = x1 + box_width
y2 = y1 + box_height

# Usar x1,2 e y1,2 para ajustes finos na máscara
yad1 = -40
yad2 = 40
xad1 = 20
xad2 = 80

marker[y1+yad1:-1-y1-yad2, x1+xad1:-1-x1-xad2] = 0

I7 = visco.imreconstruction(I6, marker)
I8 = I6-I7

# Desenha o bounding box (linha verde, espessura 2)
# Convert I8 to a BGR image to draw the rectangle
I8_bgr = cv2.cvtColor(I8, cv2.COLOR_GRAY2BGR)
cv2.rectangle(I8_bgr, (x1+xad1, y1+yad1), (x2-xad2, y2-yad2), (0, 255, 0), 2)
cv2.imshow('Image with applied mask - BB', I8_bgr)

# cv2.imshow('Imagem limiarizada com elementos retirados I6', marker)
# cv2.imshow('Imagem limiarizada com elementos retirados', I8)

cv2.waitKey(0)
infoRegioes = visco.analisaRegioes(I8)

print('Regioes encontradas: ', len(infoRegioes))
# print('Regioes: ', infoRegioes)

X = np.zeros(4, np.float32)
Y = np.zeros(4,np.float32)

# -------- BOUNDING BOX -----------

# num_labels, I_labels = cv2.connectedComponents(I8)

# index = 4
# Icomponent = np.uint8(I_labels == index) * 255

# y,x = np.where(Icomponent)

# ymin = np.min(y)
# ymax = np.max(y)
# xmin = np.min(x)
# xmax = np.max(x)

# I9 = cv2.cvtColor(I8.copy(), cv2.COLOR_GRAY2BGR)
# p1 = np.array([xmin, ymin])
# p2 = np.array([xmax, ymax])

# print(f'Ponto 1: {p1}')
# print(f'Ponto 2: {p2}')

# color = (0,0,255)
# thickness = 1

# cv2.rectangle(I9, p1, p2, color, thickness)

# cv2.imshow('Imagem com bounding box', I9)

# -------- BOUNDING BOX -----------

for k in range(len(infoRegioes)):

    x = infoRegioes[k]['centroide']
    x = np.append(x,1)
    x = x.reshape(3,1)

    print(f'Região x: {x}')

    Point = np.linalg.matmul(np.linalg.inv(P2), x)

    print(f'Point: {Point}')

    X[k] = Point[0,0]/Point[2,0]
    Y[k] = Point[1,0]/Point[2,0]

    print(f'\nIm{k+1} (X,Y): ({X[k]}, {Y[k]})\n')

    # print(f'Coordenadas no mundo real: {X[k]} e {Y[k]} em cm')


for k in range(1, len(infoRegioes)):
    # print(f'{X[0]}, {X[k]}, {Y[0]}, {Y[k]}')
    distancia = np.sqrt((X[0]-X[k])**2 + (Y[0]-Y[k])**2)
    print(f'Distancia {distancia} cm')

    # Criar uma cópia da imagem binária para desenhar as linhas
    I9 = cv2.cvtColor(I8.copy(), cv2.COLOR_GRAY2BGR)

    # Calcular as distâncias entre os centroides
    distancias = [np.sqrt((X[0] - X[k])**2 + (Y[0] - Y[k])**2) for k in range(1, len(infoRegioes))]

    # Desenhar linhas conectando os centroides e exibir as distâncias
    for k, distancia in enumerate(distancias, start=1):
        # Coordenadas dos centroides
        x1, y1 = int(infoRegioes[0]['centroide'][0]), int(infoRegioes[0]['centroide'][1])
        x2, y2 = int(infoRegioes[k]['centroide'][0]), int(infoRegioes[k]['centroide'][1])
        
        # Desenhar a linha
        cv2.line(I2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calcular a posição do texto (meio da linha)
        text_x = x2-35#(x1 + x2) // 2
        text_y = y2+15#(y1 + y2) // 2
        
        # Escrever a distância na imagem
        cv2.putText(I2, f"{distancia:.2f}cm", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Exibir a imagem com as linhas e distâncias
    cv2.imshow('Imagem com linhas e distâncias', I2)

cv2.waitKey(0)