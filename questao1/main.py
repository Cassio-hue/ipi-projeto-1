import math
import numpy as np
import matplotlib.pyplot as plt

def LER_YUV(filename, width, height, frame_num):

    num_pixels_y = width * height
    num_pixels_uv = (width // 2) * (height // 2)
    frame_size = num_pixels_y + 2 * num_pixels_uv
    offset = frame_num * frame_size
    
    with open(filename, 'rb') as file:
        file.seek(offset)

        Y = np.frombuffer(file.read(num_pixels_y), dtype=np.uint8).reshape((height, width))
        U = np.frombuffer(file.read(num_pixels_uv), dtype=np.uint8).reshape((height // 2, width // 2))
        V = np.frombuffer(file.read(num_pixels_uv), dtype=np.uint8).reshape((height // 2, width // 2))
    
    return Y, U, V

# Questão 1.2
def double (matrix):
    height, width = matrix.shape
    matrix_doubled = np.zeros((height * 2, width * 2), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            U_value = matrix[i, j]
            matrix_doubled[i*2, j*2] = U_value


    for i in range(height * 2):
        for j in range(width * 2):
            if matrix_doubled[i, j] == 0:
                matrix_doubled[i, j] = matrix_doubled[i-1, j] if i > 0 and matrix_doubled[i-1, j] != 0 else matrix_doubled[i, j-1] if j > 0 else 0
                # matrix_doubled[i, j] = interpolate_value(matrix_doubled, i, j)

    return matrix_doubled

# Questão 1.3
def interpolate_value(U_upsampled, i, j):
    values = []
    weights = []
    
    # Valor superior
    if i > 0 and U_upsampled[i-1, j] != 0:
        values.append(U_upsampled[i-1, j])
        weights.append(1)
    
    # Valor à esquerda
    if j > 0 and U_upsampled[i, j-1] != 0:
        values.append(U_upsampled[i, j-1])
        weights.append(1)
    
    if values:
        return math.ceil(sum(values) / len(values))
    else:
        return 0


def YUV420_to_RGB(Y, U, V):

    YUV = np.stack((Y, U, V), axis=-1).astype(np.float32)
    YUV[:, :, 0] -= 16
    YUV[:, :, 1:] -= 128

    RGB = np.dot(YUV, np.array([[1.164,  0.000,  1.596],
                            [1.164, -0.392, -0.813],
                            [1.164,  2.017,  0.000]]).T)
    
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    
    return RGB

# Ler a imagem
Y, U, V = LER_YUV('foreman.yuv', 352, 288, 10)

height, width = Y.shape

U_x1 = double(U)
V_x1 = double(V)
Y_x1 = double(Y)
# RGB = YUV420_to_RGB(Y, U_x1, V_x1)

U_x2 = double(U_x1)
V_x2 = double(V_x1)
RGB = YUV420_to_RGB(Y_x1, U_x2, V_x2)

plt.imshow(RGB)
plt.title('Quadro YUV convertido para RGB')
plt.axis('off') 
plt.show()
