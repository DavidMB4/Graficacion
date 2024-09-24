import numpy as np
import cv2

def generar_punto_elipse(a, b, t):
    x = int(a * np.cos(t) + 300)  
    y = int(b * np.sin(t) + 300)
    return (x, y)

img_width, img_height = 600, 600

mercurio_ejeMayor = 180  
mercurio_ejeMenor = 100 
num_puntos = 1000

venus_ejeMayor = 200
venus_ejeMenor = 120

t_vals = np.linspace(0, 2 * np.pi, num_puntos)

for t in t_vals:

    imagen = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    t_mercurio = t * 1 
    t_venus = t * 0.8  

    mercurioPunto = generar_punto_elipse(mercurio_ejeMayor, mercurio_ejeMenor, t_mercurio)
    veunsPunto = generar_punto_elipse(venus_ejeMayor, venus_ejeMenor, t_venus)
    

    cv2.circle(imagen, mercurioPunto, radius=5, color=(126, 200, 243), thickness=-1)
    cv2.circle(imagen, veunsPunto, radius=10, color=(0, 100, 158), thickness=-1)
    

    for t_tray in t_vals:
        pt_tray_mercurio = generar_punto_elipse(mercurio_ejeMayor, mercurio_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_mercurio, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_Venus = generar_punto_elipse(venus_ejeMayor, venus_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_Venus, radius=1, color=(255, 255, 255), thickness=-1)
    

    cv2.imshow('img', imagen)
    

    cv2.waitKey(10)
    
cv2.destroyAllWindows()