import cv2 as cv
import numpy as np
import math

img=cv.imread('C:/Users/david/OneDrive/Documentos/Tec/5 Semestre/Graficacion/Repositorio/Actividad1/pokeball.png', 0)
x, y=img.shape
center_X, center_y = (x // 2, y // 2)
angle = 70
scale_x, scale_y = 2, 2
dx, dy = 20, 20

img2 = np.zeros((int(x*2.5),int(y*2.5)), dtype=np.uint8)
xx, yy = img2.shape
theta = math.radians(angle)

for i in range(x):
    for j in range(y):
        
        x2 = int((int((j - center_X) * math.cos(theta) - (i - center_y) * math.sin(theta) + center_X))*scale_x)+dx
        y2 = int((int((j - center_X) * math.sin(theta) + (i - center_y) * math.cos(theta) + center_y))*scale_y)+dy
        if 0 <= x2 < int(y*2.5) and 0 <= y2 < int(x*2.5):
            img2[y2, x2] = img[i, j]
        
cv.imshow('img', img)
cv.imshow('img2', img2)
cv.waitKey()
cv.destroyAllWindows()