import cv2 as cv
import numpy as np
import math

img=cv.imread('C:/Users/david/OneDrive/Documentos/Tec/5 Semestre/Graficacion/Repositorio/Actividad1/pokeball.png', 0)
x, y=img.shape
center_X, center_y = (x // 2, y // 2)
angle = 60
scale_x, scale_y = 0.5, 0.5
dx, dy = 10, 10

img2 = np.zeros((x, y), dtype=np.uint8)
theta = math.radians(angle)

for i in range(x):
    for j in range(y):
        
        x2 = int((int((j - center_X) * math.cos(theta) - (i - center_y) * math.sin(theta) + center_X))*scale_x)+dx
        y2 = int((int((j - center_X) * math.sin(theta) + (i - center_y) * math.cos(theta) + center_y))*scale_y)+dy
        if 0 <= x2 < y and 0 <= y2 < x:
            img2[y2, x2] = img[i, j]
        
cv.imshow('img', img)
cv.imshow('img2', img2)
cv.waitKey()
cv.destroyAllWindows()