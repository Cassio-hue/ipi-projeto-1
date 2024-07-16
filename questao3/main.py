import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_reject_filter(rows, cols, d0, uk, vk):
    filter = np.zeros((rows, cols))
    
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u - rows / 2 + uk) ** 2 + (v - cols / 2 + vk) ** 2)
            D_muv = np.sqrt((u - rows / 2 - uk) ** 2 + (v - cols / 2 - vk) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                filter[u, v] = 0.0
            else:
                filter[u, v] = 1.0
    
    return filter

# Carregar a imagem em escala de cinza
src = cv2.imread('moire.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = src.shape

# Aplicar a FFT
dft = np.fft.fft2(src)
dft_shift = np.fft.fftshift(dft)

# Obter a magnitude do espectro para visualização
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Criar filtros Butterworth
# 1° PAR: D0=10, uk=39, vk=30
# 2° PAR: D0=10, uk=-39, vk=30
# 3° PAR: D0=5, uk=78, vk=30
# 4° PAR: D0=5, uk=-78, vk=30
H1 = notch_reject_filter(rows, cols, 10, 39, 30)
H2 = notch_reject_filter(rows, cols, 10, -39, 30)
H3 = notch_reject_filter(rows, cols, 5, 78, 30)
H4 = notch_reject_filter(rows, cols, 5, -78, 30)

# Multiplicar os filtros
NotchFilter = H1*H2*H3*H4
NotchRejectCenter = dft_shift * NotchFilter
NotchReject = np.fft.ifftshift(NotchRejectCenter)
inverse_NotchReject = np.fft.ifft2(NotchReject) 
Result = np.abs(inverse_NotchReject)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(src, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Espectro de Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(Result, cmap='gray')
plt.title('Imagem Filtrada')
plt.axis('off')

plt.show()
