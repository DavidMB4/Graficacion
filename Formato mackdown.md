# Repositorio graficación

### Actividad pixelart

~~~
import numpy as np
import cv2

pixel = [
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    # Cabeza
    [255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,0,90,90,90,90,90,90,90,90,90,90,90,90,0,0,0,90,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,0,90,90,90,0,165,90,90,90,90,90,90,90,90,90,90,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,0,90,90,0,165,165,90,90,90,90,90,90,90,90,90,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,0,90,0,165,90,90,90,90,90,90,90,90,90,90,90,0,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,0,90,90,90,90,90,90,90,90,255,255,255,90,90,90,90,0,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,0,90,90,90,90,90,90,90,90,255,255,255,255,255,90,90,90,0,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,0,90,90,90,90,90,90,90,90,90,255,255,255,255,0,90,255,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,0,0,0,0,0,0,90,90,90,90,90,255,255,255,255,0,90,255,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,0,90,90,90,90,90,90,90,255,255,255,0,255,0,0,0,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,0,90,90,90,90,90,165,165,165,255,255,255,255,165,165,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,0,90,90,90,90,90,90,165,165,165,165,165,165,165,165,165,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,0,0,0,0,0,0,0,0,0,165,165,165,165,165,165,165,0,255,255,255,255,255,255,255,255,255,255],
    # Torso y brazos
    [255,255,255,255,255,255,255,255,255,255,255,255,0,90,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,0,165,0,0,90,165,165,165,0,0,0,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,0,165,0,255,255,0,165,165,165,0,255,255,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,255,0,165,165,0,255,255,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,0,255,255,0,0,90,0,0,0,0,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,90,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255],
    #Piernas
    [255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,90,0,110,110,0,0,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,0,0,110,110,0,0,110,110,255,0,0,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,0,110,110,110,255,110,0,0,110,110,110,0,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,0,110,110,255,110,110,110,110,0,110,110,110,0,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
    [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
]


pixel_array = np.array(pixel, dtype=np.uint8)

escalado = cv2.resize(pixel_array, (pixel_array.shape[1] * 10, pixel_array.shape[0] * 10), interpolation=cv2.INTER_NEAREST)

cv2.imshow('Pixel Art', escalado)
cv2.waitKey(0)
cv2.destroyAllWindows()

~~~

### Actividad transformaciones geometricas

##### Parte 1

~~~
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
~~~

##### Parte 2
~~~
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

~~~

###### Parte 3
~~~
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

~~~
![Imagen png usada para el ejercicio](https://github.com/DavidMB4/Graficacion/blob/master/Actividad1Transformaciones/pokeball.png)


###