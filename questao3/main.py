import cv2
import numpy as np
from matplotlib import pyplot as plt

def butterworth_high_pass(size, cutoff, order, notch_center):
    rows, cols = size
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow - notch_center[0])**2 + (v - ccol - notch_center[1])**2)
            D_dash = np.sqrt((u - crow + notch_center[0])**2 + (v - ccol + notch_center[1])**2)
            mask[u, v] = 1 / (1 + (cutoff / D)**(2 * order)) * 1 / (1 + (cutoff / D_dash)**(2 * order))
    return mask
    

def apply_notch_filter(img, cutoff, order, notch_centers):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    mask = np.ones((rows, cols, 2), np.float32)
    for notch_center in notch_centers:
        high_pass = butterworth_high_pass((rows, cols), cutoff, order, notch_center)
        mask[:, :, 0] *= high_pass
        mask[:, :, 1] *= high_pass

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back

# Carregar imagem em escala de cinza
img = cv2.imread('moire.tif', 0)

# Definir parâmetros do filtro
cutoff = 30
order = 4
notch_centers = [(30, 30), (-30, -30)]  # Exemplo de centros de notch

# Aplicar filtro rejeita-notch
filtered_img = apply_notch_filter(img, cutoff, order, notch_centers)

# Mostrar imagem original e filtrada
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Imagem Original')
plt.subplot(122), plt.imshow(filtered_img, cmap='gray'), plt.title('Imagem Filtrada')
plt.show()
