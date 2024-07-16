import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def applyFilter(src, kernel, c=None):
    if c == 4 or c == -4:
      # desvio padrão
      dp = 1
      src = cv.GaussianBlur(src, (3, 3), dp)

    depth = -1
    dst = cv.filter2D(src, depth, kernel)

    return cv.convertScaleAbs(dst)

def add_label(image, label):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 0, 255)
    font_thickness = 2

    text_size = cv.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 10
    
    cv.putText(image, label, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv.LINE_AA)

# Carregar imagem
imageName = 'Image1.pgm'
src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR)

dictionary = {
    'kernel-4-negative': np.array([
        [ 0,  1, 0],
        [ 1, -4, 1],
        [ 0,  1, 0]], dtype=np.float32),
    'kernel-4-positive': np.array([        
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]], dtype=np.float32),
    'kernel-8-negative': np.array([
        [ 1,  1, 1],
        [ 1, -8, 1],
        [ 1,  1, 1]], dtype=np.float32),
    'kernel-8-positive': np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]], dtype=np.float32),
}

array = []

for key in dictionary:
    if '4' in key:
        c = 4
        if 'negative' in key:
            c = -4
    if '8' in key:
        c = 8
        if 'negative' in key:
            c = -8

    filteredImage = applyFilter(src, dictionary[key], c)
    
    if key.endswith('positive'):
        abs_dst = cv.add(src, filteredImage)
    else:
        filteredImage = filteredImage * -1
        abs_dst = src + filteredImage

    # Adicionar a chave do dicionário como legenda
    add_label(abs_dst, key)
    array.append(abs_dst)

top_row = np.hstack((array[0], array[1]))
bottom_row = np.hstack((array[2], array[3]))

combined_image = np.vstack((top_row, bottom_row))

plt.imshow(src)
plt.imshow(combined_image)
plt.axis('off')
plt.show()
