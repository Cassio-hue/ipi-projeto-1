import numpy as np
import matplotlib.pyplot as plt

def LER_YUV(filename, width, height, start_frame, num_frames):
    num_pixels_y = width * height
    num_pixels_uv = (width // 2) * (height // 2)
    frame_size = num_pixels_y + 2 * num_pixels_uv
    offset = start_frame * frame_size
    frames = []

    with open(filename, 'rb') as file:
        file.seek(offset)
        for _ in range(num_frames):
            Y = np.frombuffer(file.read(num_pixels_y), dtype=np.uint8).reshape((height, width))
            U = np.frombuffer(file.read(num_pixels_uv), dtype=np.uint8).reshape((height // 2, width // 2))
            V = np.frombuffer(file.read(num_pixels_uv), dtype=np.uint8).reshape((height // 2, width // 2))
            frames.append((Y, U, V))
    
    return frames

def upsample(U, V, width, height):
    U_upsampled = np.zeros((height, width), dtype=np.uint8)
    V_upsampled = np.zeros((height, width), dtype=np.uint8)

    for i in range(height // 2):
        for j in range(width // 2):
            U_value = U[i, j]
            V_value = V[i, j]
            U_upsampled[i*2:i*2+2, j*2:j*2+2] = U_value
            V_upsampled[i*2:i*2+2, j*2:j*2+2] = V_value
    
    return U_upsampled, V_upsampled

def YUV420_to_RGB(Y, U, V):
    height, width = Y.shape
    U, V = upsample(U, V, width, height)
    # U, V = upsample(U, V, width, height)

    # YUV to RGB conversion
    YUV = np.stack((Y, U, V), axis=-1).astype(np.float32)
    YUV[:, :, 0] -= 16
    YUV[:, :, 1:] -= 128

    RGB = np.dot(YUV, np.array([[1.164,  0.000,  1.596],
                                [1.164, -0.392, -0.813],
                                [1.164,  2.017,  0.000]]).T)
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    
    return RGB

# Exemplo de uso
frames = LER_YUV('foreman.yuv', 352, 288, 10, 1)
Y, U, V = frames[0]
RGB = YUV420_to_RGB(Y, U, V)

# Mostra o quadro RGB usando matplotlib
plt.imshow(RGB)
plt.title('Quadro YUV convertido para RGB')
plt.axis('off')  # Oculta os eixos
plt.show()
