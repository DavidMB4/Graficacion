import cv2
import numpy as np
import math

ancho, alto = 500, 500
imagen = 255 * np.ones((alto, ancho, 3), np.uint8)

inicioX, inicioY = 200, 200
velX, velY = 5, 7

while True:
    imagen = 255 * np.ones((alto, ancho, 3), np.uint8)
    
    cv2.circle(imagen, (inicioX,inicioY), 20, (0, 0, 255), -1)
    
    cv2.imshow('imagen', imagen)
    
    inicioX = inicioX + velX
    inicioY = inicioY + velY
    
    if inicioX - 20 <= 0 or inicioX + 20 >= ancho:
        velX = -velX
    if inicioY - 20 <= 0 or inicioY + 20 >= alto:
        velY = -velY
        
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()
