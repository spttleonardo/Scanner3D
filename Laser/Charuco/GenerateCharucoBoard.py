import os
import numpy as np
import cv2

# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT           = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY   = 7
SQUARES_HORIZONTALLY = 7
SQUARE_LENGTH        = 0.025   # 3 cm
MARKER_LENGTH        = 0.0167  # 1.5 cm (~2/3 do quadrado)
PAGE_WIDTH_PX        = 640    # largura da página em pixels
MARGIN_PX            = 20     # margem de cada lado em pixels
SAVE_NAME            = 'ChArUco_Marker_A4.png'
# ------------------------------

def create_and_save_new_board():
    # calcula altura da página A4 mantendo proporção 210mm×297mm
    page_height = int(PAGE_WIDTH_PX * 297/210)

    # 1) cria o CharucoBoard com a API antiga
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
                                    SQUARE_LENGTH, MARKER_LENGTH, dictionary)

    # 2) define área útil do tabuleiro (sem margens externas)
    board_w = PAGE_WIDTH_PX - 2 * MARGIN_PX
    board_h = int(board_w * (SQUARES_VERTICALLY / SQUARES_HORIZONTALLY))

    # 3) gera a imagem do tabuleiro (sem margem interna)
    board_img = cv2.aruco.CharucoBoard.generateImage(board,
                            (board_w, board_h), marginSize=0)

    # 4) cria a página A4 em branco (canal único, valores 0–255)
    page = np.ones((page_height, PAGE_WIDTH_PX), dtype=np.uint8) * 255

    # 5) centraliza o board na página (horizontalmente e verticalmente)
    x0 = MARGIN_PX
    y0 = (page_height - board_h) // 2
    page[y0:y0+board_h, x0:x0+board_w] = board_img

    # 6) exibe e salva
    cv2.imshow("ChArUcoBoard A4", page)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    cv2.imwrite(SAVE_NAME, page)
    print(f"✔️ CharucoBoard salvo em '{SAVE_NAME}' ({PAGE_WIDTH_PX}×{page_height} px)")
    print("→ Imprima em A4 sem escalonar para quadrados de 3 cm de lado.")

if __name__ == '__main__':
    create_and_save_new_board()
