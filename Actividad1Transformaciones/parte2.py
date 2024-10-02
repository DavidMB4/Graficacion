import cv2 as cv
import numpy as np
import math

img = cv.imread('C:/Users/david/OneDrive/Documentos/Tec/5 Semestre/Graficacion/Repositorio/Actividad1/pokeball.png', 0)
x, y = img.shape
center_X, center_y = (x // 2, y // 2)

angle_horario = 30
angle_antihorario = -60

scale_x, scale_y = 2, 2

img2 = np.zeros((int(x*2.5), int(y*2.5)), dtype=np.uint8)
img3 = np.zeros((int(x*2.5), int(y*2.5)), dtype=np.uint8)

theta_horario = math.radians(angle_horario)
theta_antihorario = math.radians(angle_antihorario)

for i in range(x):
    for j in range(y):
        x2 = int((int((j - center_X) * math.cos(theta_horario) - (i - center_y) * math.sin(theta_horario) + center_X))*scale_x)
        y2 = int((int((j - center_X) * math.sin(theta_horario) + (i - center_y) * math.cos(theta_horario) + center_y))*scale_y)
        if 0 <= x2 < img2.shape[1] and 0 <= y2 < img2.shape[0]:
            img2[y2, x2] = img[i, j]

x_new, y_new = img2.shape
center_X_new, center_y_new = (x_new // 2, y_new // 2)

for i in range(x_new):
    for j in range(y_new):
        if img2[i, j] > 0:  
            x3 = int((int((j - center_X_new) * math.cos(theta_antihorario) - (i - center_y_new) * math.sin(theta_antihorario) + center_X_new)))
            y3 = int((int((j - center_X_new) * math.sin(theta_antihorario) + (i - center_y_new) * math.cos(theta_antihorario) + center_y_new)))
            if 0 <= x3 < img3.shape[1] and 0 <= y3 < img3.shape[0]:
                img3[y3, x3] = img2[i, j]

cv.imshow('img original', img)
cv.imshow('img2 30 grados horario', img2)
cv.imshow('img3 60 grados antihorario', img3)
cv.waitKey()
cv.destroyAllWindows()
