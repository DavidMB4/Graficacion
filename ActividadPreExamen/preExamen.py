import cv2 as cv
import numpy as np

imagen = cv.imread('salida.png', 1)
imagenHSV = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)

rangoRojoBajo = np.array([0, 40, 40])
rangoRojoAlto = np.array([10, 255, 255])

rangoRojoBajo2 = np.array([160, 40, 40])
rangoRojoAlto2 = np.array([180, 255, 255])

mascaraRojo1 = cv.inRange(imagenHSV, rangoRojoBajo, rangoRojoAlto)
mascaraRojo2 = cv.inRange(imagenHSV, rangoRojoBajo2, rangoRojoAlto2)
mascaraRojo = cv.add(mascaraRojo1, mascaraRojo2)

rangoAzulBajo = np.array([85, 40, 40])
rangoAzulAlto = np.array([128, 255, 255])

mascaraAzul = cv.inRange(imagenHSV, rangoAzulBajo, rangoAzulAlto)

rangoVerdeBajo = np.array([40, 40, 40])
rangoVerdeAlto = np.array([80, 255, 255])

mascaraVerde = cv.inRange(imagenHSV, rangoVerdeBajo, rangoVerdeAlto)

rangoAmarilloBajo = np.array([22, 40, 40])
rangoAmarilloAlto = np.array([32, 40, 40])

mascaraAmarillo = cv.inRange(imagenHSV, rangoAmarilloBajo, rangoAmarilloAlto)

alto, ancho, canales = imagen.shape
print(alto)
print(ancho)

for y in range(alto):
    for x in range(ancho):
        if mascaraRojo[y, x] == 0:
            mascaraRojo[y, x] = 150
        else:
            mascaraRojo[y, x] = 180
            
def recorre(x, y):
    if ()

            
print(y)
print(x)
        

cv.imshow('EJEMPLO', mascaraRojo)
cv.waitKey(0)
cv.destroyAllWindows()