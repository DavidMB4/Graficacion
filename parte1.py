import cv2 as cv
import numpy as np
import math

img=cv.imread('C:/Users/david/OneDrive/Documentos/Tec/5 Semestre/Graficacion/Dibujo pixel/pokeball.png', 0)
x, y=img.shape
center_X, center_y = (x // 2, y // 2)
angle = 60
shear_factor_x = 0.5
dx, dy = 10, 10

img2 = np.zeros((int(x * shear_factor_x), int(y * shear_factor_x)), dtype=np.uint8)
xx, yy = img2.shape
theta = math.radians(angle)

for i in range(x):
    for j in range(y):
        img2[int(i*shear_factor_x)+dx,int(j*shear_factor_x)+dy]=img[i, j] 
        
cv.imshow('img', img)
cv.imshow('img2', img2)
cv.waitKey()
cv.destroyAllWindows()