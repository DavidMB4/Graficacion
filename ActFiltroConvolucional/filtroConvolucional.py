import numpy as np
import cv2

imagen = cv2.imread('girasol.jpg',1)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
filas, columnas = len(imagen_gris), len(imagen_gris[0])

#Guia para la posición
y=1
x=1

matriz_con_borde = np.zeros((filas + 2, columnas + 2))

resultado = np.zeros((filas, columnas), dtype=np.uint8)

for i in range(filas):
    for j in range(columnas):
        matriz_con_borde[i + 1][j + 1] = imagen_gris[i][j]

#Filtro 1/9 para la matriz
for i in range(filas):
    for j in range(columnas):
        resultado[i][j] = matriz_con_borde[x+i-1,y+j-1]*(1/9)+matriz_con_borde[x+i,y+j-1]*(1/9)+matriz_con_borde[x+i+1,y+j-1]*(1/9)+matriz_con_borde[x+i-1,y+j]*(1/9)+matriz_con_borde[x+i,y+j]*(1/9)+matriz_con_borde[x+i+1,y+j]*(1/9)+matriz_con_borde[x+i-1,y+j+1]*(1/9)+matriz_con_borde[x+i,y+j+1]*(1/9)+matriz_con_borde[x+i+1,y+j+1]*(1/9)
        
cv2.imshow("Imagen con filtro", resultado)
cv2.imshow("Imagen original", imagen_gris)

cv2.waitKey(0)
cv2.destroyAllWindows()