import cv2
import numpy as np

imagen = cv2.imread('teoriaColor.jpg', 1)

imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Imagen color verde
bajo_verde = np.array([40, 40, 40])
alto_verde = np.array([80, 255, 255])

mascara_verde = cv2.inRange(imagen_hsv, bajo_verde, alto_verde)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

resultadoVerde = np.where(mascara_verde[:, :, None] == 255, imagen, imagen_gris_bgr)

#Imagen color Azul
bajo_azul = np.array([85, 40, 40])
alto_azul = np.array([129, 255, 255])

mascara_azul = cv2.inRange(imagen_hsv, bajo_azul, alto_azul)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

resultadoAzul = np.where(mascara_azul[:, :, None] == 255, imagen, imagen_gris_bgr)

#Imagen color Amarillo
bajo_amarillo = np.array([22, 40, 40])
alto_amarillo = np.array([32, 255, 255])

mascara_amarillo = cv2.inRange(imagen_hsv, bajo_amarillo, alto_amarillo)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

resultadoAmarillo = np.where(mascara_amarillo[:, :, None] == 255, imagen, imagen_gris_bgr)

#Imagen color Morado
bajo_morado = np.array([130, 40, 40])
alto_morado = np.array([145, 255, 255])

mascara_morado = cv2.inRange(imagen_hsv, bajo_morado, alto_morado)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

resultadoMorado = np.where(mascara_morado[:, :, None] == 255, imagen, imagen_gris_bgr)


#Imagen color Rosado
bajo_rosa = np.array([146, 40, 40])
alto_rosa = np.array([162, 255, 255])

mascara_rosa = cv2.inRange(imagen_hsv, bajo_rosa, alto_rosa)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

resultadoRosa = np.where(mascara_rosa[:, :, None] == 255, imagen, imagen_gris_bgr)

cv2.imshow('Color verde resaltado', resultadoVerde)
cv2.imshow('Color azul resaltado', resultadoAzul)
cv2.imshow('Color amarillo resaltado', resultadoAmarillo)
cv2.imshow('Color morado resaltado', resultadoMorado)
cv2.imshow('Color rosado resaltado', resultadoRosa)
cv2.imshow('Original', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()