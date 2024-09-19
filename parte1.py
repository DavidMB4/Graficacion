import cv2 as cv
import numpy as np

img=cv.imread('pokeball.png', 0)
x, y=img.shape
center = (y // 2, x // 2)
angle = 60
shear_factor_x = 0.5
dx, dy = 10, 10

img2 = np.zeros((int(x * shear_factor_x), int(y * shear_factor_x)), dtype=np.uint8)

for i in range(x):
    for j in range(y):
        img2[int(i*shear_factor_x)+dx,int(j*shear_factor_x)+dy]=img[i, j] 