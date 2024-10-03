import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    if ret:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #Imagen color Azul
        bajo_azul = np.array([85, 40, 40])
        alto_azul = np.array([129, 255, 255])
        imagen_gris = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
        imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
        
        
        maskColor = cv2.inRange(hsv, bajo_azul, alto_azul)
        img_azul = cv2.bitwise_and(img, img, mask=maskColor)
        resultadoAzul = np.where(maskColor[:, :, None] == 255, img_azul, imagen_gris_bgr)
        cv2.imshow('video', resultadoAzul)
        
        k =cv2.waitKey(1) & 0xFF
        if k == 27 :
            break
    else:
        break