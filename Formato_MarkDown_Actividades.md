# Repositorio graficación

### Actividad pixelart

Es una matriz que tiene valores de 0 a 255 para poder tomar una escala de grise, como son pocos los pixeles se mostraria una imagen muy pequeña, asi que se usa:
~~~
cv2.resize(pixel_array, (pixel_array.shape[1] * 10, pixel_array.shape[0] * 10), interpolation=cv2.INTER_NEAREST)
~~~
Para hacer un escalado y que la imagen se vea mejor.

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

En esta imagen se hace una rotacion de 60 grados en sentido horario, luego se escala a 0.5 lo que hace que la imagen sea un poco mas pequeña, y por ultimo se mmueve en X y Y 10 pixeles.

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
En esta imagen se escala en 2 lo que significa que es el doble de su tamaño original ya que se multiplica, luego se hace una rotacion en sentido horario a 30 grados, y luego otra en sentido antihorario de 60 grados. Imprimiendo 2 imagenes, una con 30 grados en sentido horario y la otra aplicandole 60 grados en sentido antihorario a la primera.

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

En esta imagen se rota en 70 grados en sentido horario, se escala en 2 obteniendo el doble de su tamaño y se traslada 20 pixeles en X y Y.
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

Imagen usada para el ejercicio
![Imagen png usada para el ejercicio](https://github.com/DavidMB4/Graficacion/blob/master/Actividad1Transformaciones/pokeball.png?raw=true)


### Actividad dibujo con primitivas

Para hacer una imagen de arboles con manzanas en unas colinas se usan circulos para representar o darle forma a las hojas de los arboles, a las colinas, a las manzanas, se usan 3 ciruclos balncos para las nubes, y para representar el sol. Mientras que para el tronco de los arboles se usan rectangulos cafes.
~~~
import cv2 as cv
import numpy as np

img = np.ones((500, 500, 3),dtype=np.uint8)*255 

cv.rectangle(img, (1,1), (500,500), (237, 160, 21), -1)

cv.circle(img, (430, 70), 50, (13, 201, 231), -1)
cv.circle(img, (250, 555), 180, (2, 200, 11), -1)
cv.circle(img, (70, 550), 180, (2, 171, 10), -1)
cv.circle(img, (420, 550), 180, (12, 193, 135), -1)
cv.rectangle(img, (45,400), (70,350), (1, 71, 109 ), -1)
cv.rectangle(img, (240,400), (270,335), (1, 71, 109 ), -1)
cv.rectangle(img, (420,400), (450,340), (1, 71, 109 ), -1)
cv.circle(img, (58, 340), 35, (0, 255, 174), -1)
cv.circle(img, (253, 340), 35, (0, 255, 174), -1)
cv.circle(img, (433, 340), 35, (0, 255, 174), -1)
cv.circle(img, (40, 340), 5, (8, 48, 231), -1)
cv.circle(img, (78, 330), 5, (8, 48, 231), -1)
cv.circle(img, (230, 330), 5, (8, 48, 231), -1)
cv.circle(img, (270, 330), 5, (8, 48, 231), -1)
cv.circle(img, (410, 330), 5, (8, 48, 231), -1)
cv.circle(img, (450, 355), 5, (8, 48, 231), -1)

cv.circle(img, (60, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (80, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (100, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (200, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (220, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (240, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (340, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (360, 150), 20, (240, 240, 240 ), -1)
cv.circle(img, (380, 150), 20, (240, 240, 240 ), -1)

cv.imshow('img', img)
cv.waitKey()
cv.destroyAllWindows()
~~~

### Actividad investigar ecuaciones parametricas
#### ¿Qué es una ecuación paramétrica?

Una ecuación paramétrica es una forma de mapear múltiples variables en una sola variable. Por tanto, en lugar de coordenadas \( x \) y \( y \), utilizamos coordenadas \( t \).

Por ejemplo:
Tenemos \( y = x^2 \), que es una ecuación cartesiana con coordenadas \( (x, y) \). Podemos cambiarla, suponiendo que \( x = t \) y \( y = t^2 \). Ahora tenemos las coordenadas \( (t, t^2) \). Esto está en forma paramétrica.

##### Ecuación paramétrica de la circunferencia

Las circunferencias tienen una ecuación general cartesiana. Esta ecuación es \( (x - a)^2 + (y - b)^2 = r^2 \), para una circunferencia con centro \( C(a, b) \) y radio \( r \).
Una identidad clave que podemos usar al elevar al cuadrado dos valores importantes diferentes es la identidad de la circunferencia unitaria: 
\[
\sin^2(\theta) + \cos^2(\theta) = 1
\]

##### Ecuación paramétrica de la recta

Para expresar la ecuación de una recta, usando un parámetro, se requieren dos cosas:
- Un punto sobre la recta
- Un vector

##### Ecuación paramétrica de la elipse

Una elipse con centro en \( (x_0, y_0) \), que se interseca con el eje \( x \) en \( (x_0 \pm a, 0) \), y con el eje \( y \) en \( (0, y_0 \pm b) \), verifica que:
\[
\frac{(x - x_0)^2}{a^2} + \frac{(y - y_0)^2}{b^2} = 1
\]

Una expresión paramétrica es:
![Imagen ecuación paramétrica elipse](https://github.com/DavidMB4/Graficacion/blob/master/InvestEcParametricas/ecuaicon%20parametrica%20elipse.jpg?raw=true)

##### Otras curvas

La expresión paramétrica de una función permite la construcción de una gran variedad de formas, simplemente variando alguna constante.
![Imagen otras curvas](https://github.com/DavidMB4/Graficacion/blob/master/InvestEcParametricas/otras%20curvas.jpg?raw=true)
que, para la cual, dependiendo del ratio a/b pueden obtenerse formas muy diversas.

### Actividad hacer 10 ecuaciones parametricas

Para hacer las primeras 9 se usan diferentes valores para k:
~~~
import numpy as np
import cv2

width, height = 1000, 1000  
img = np.ones((height, width, 3), dtype=np.uint8)*255

a, b = 150, 100  
k = 0.1608
theta_increment = 0.05  
max_theta = 2 * np.pi 

center_x, center_y = width // 2, height // 2

theta = 0  

while True:  
    img = np.ones((width, height, 3), dtype=np.uint8) * 255
    
    for t in np.arange(0, theta, theta_increment):
        r = a + b * np.cos(k * t)
        x = int(center_x + r * np.cos(t))
        y = int(center_y + r * np.sin(t))
        
        cv2.circle(img, (x, y), 2, (209, 122, 2), 2) 
        cv2.circle(img, (x+2, y+2), 2, (0, 0, 0), 2)  

    cv2.imshow("Parametric Animation", img)
    
    theta += theta_increment

    if cv2.waitKey(30) & 0xFF == 27: 
        break

cv2.destroyAllWindows()
~~~
Imagen resultado de ecuacion con k=2.5
![Imagen resultado de ecuacion con k=2.5](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen1%20k=2.5.jpg?raw=true)

Imagen resultado de ecuacion con k=7.5
![Imagen resultado de ecuacion con k=7.5](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen2%20k=7.5.jpg?raw=true)

Imagen resultado de ecuacion con k=1.21
![Imagen resultado de ecuacion con k=1.21](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen3%20k=1.21.jpg?raw=true)

Imagen resultado de ecuacion con k=0.25
![Imagen resultado de ecuacion con k=0.25](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen4%20k=0.25.jpg?raw=true)

Imagen resultado de ecuacion con k=0.89
![Imagen resultado de ecuacion con k=0.89](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen5%20k=0.89.jpg?raw=true)

Imagen resultado de ecuacion con k=100.5
![Imagen resultado de ecuacion con k=100.5](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen6%20k=100.5.jpg?raw=true)

Imagen resultado de ecuacion con k=2.876
![Imagen resultado de ecuacion con k=2.876](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen7%20k=2.876.jpg?raw=true)

Imagen resultado de ecuacion con k=7.321
![Imagen resultado de ecuacion con k=7.321](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen8%20k=7.321.jpg?raw=true)

Imagen resultado de ecuacion con k=0.1608
![Imagen resultado de ecuacion con k=0.1608](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen9%20k=0.1608.jpg?raw=true)

Para la ultima imagen (10) se modifico la ecuacion completa 

~~~
    r = a + b * t
    x = int(center_x + a * np.sin(3 * t + np.pi / 2))
    y = int(center_y + b * np.sin(2 * t))
~~~
Imagen con diferente ecuacion:
![Imagen con diferente ecuacion](https://github.com/DavidMB4/Graficacion/blob/master/ActividadFormasParametricas/Imagen10%20formula%20diferente.jpg?raw=true)

~~~
import numpy as np
import cv2

width, height = 1000, 1000  
img = np.ones((height, width, 3), dtype=np.uint8)*255

a, b = 250, 200  
k = 5.5
theta_increment = 0.05  
max_theta = 2 * np.pi 

center_x, center_y = width // 2, height // 2

theta = 0  

while True:  
    img = np.ones((width, height, 3), dtype=np.uint8) * 255
    
    for t in np.arange(0, theta, theta_increment):
        r = a + b * t
        x = int(center_x + a * np.sin(3 * t + np.pi / 2))
        y = int(center_y + b * np.sin(2 * t))

        
        cv2.circle(img, (x, y), 2, (4, 193, 5), 2) 
        cv2.circle(img, (x+2, y+2), 2, (0, 0, 0), 2)  

    cv2.imshow("Parametric Animation", img)
    
    theta += theta_increment

    if cv2.waitKey(30) & 0xFF == 27: 
        break

cv2.destroyAllWindows()
~~~

### Actividad acomodar mascara de gas
Para acomodadr la mascara mejor en la camara usé:
~~~
traslacion_x = 10
traslacion_y = 25
~~~

En la region de interes (ROI) del frame
~~~
# Crear una región de interés (ROI) en el frame donde colocaremos la máscara
        roi = frame[y - traslacion_y:y - traslacion_y + h + movimientoY, x:x + w + movimientoX]
~~~

~~~
import cv2
import numpy as np

# Cargar la máscara que deseas agregar (asegúrate de que sea PNG con transparencia)
mascara = cv2.imread('gas.png', cv2.IMREAD_UNCHANGED)  # Cargar PNG con transparencia

# Cargar el clasificador preentrenado de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Capturar video desde la cámara (o puedes usar un archivo de video)
video = cv2.VideoCapture(0)  # Cambia el 0 por la ruta de un archivo de video si quieres usar un archivo

while True:
    # Leer cada frame del video
    ret, frame = video.read()

    if not ret:
        break

    # Convertir el frame a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los rostros en el frame
    rostros = face_cascade.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Procesar cada rostro detectado
    for (x, y, w, h) in rostros:
        movimientoX = 40
        movimientoY = 50 
        
        # Redimensionar la máscara para que coincida con el tamaño del rostro detectado
        mascara_redimensionada = cv2.resize(mascara, (w + movimientoX, h + movimientoY))

        # Separar los canales de la máscara: color y alfa (transparencia)
        mascara_rgb = mascara_redimensionada[:, :, :3]  # Canal de color
        mascara_alpha = mascara_redimensionada[:, :, 3]  # Canal de transparencia
        
        # Convertir mascara_alpha a tipo uint8 si no lo es
        if mascara_alpha.dtype != np.uint8:
            mascara_alpha = mascara_alpha.astype(np.uint8)
            
        #Para acomodar mascara de gas
        traslacion_x = 10
        traslacion_y = 25

        # Crear una región de interés (ROI) en el frame donde colocaremos la máscara
        roi = frame[y - traslacion_y:y - traslacion_y + h + movimientoY, x:x + w + movimientoX]

        # Verificar que el ROI coincide con el tamaño de la máscara para evitar errores
        if roi.shape[:2] == mascara_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            mascara_alpha_inv = cv2.bitwise_not(mascara_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi, roi, mask=mascara_alpha_inv)

            # Enmascarar la máscara RGB
            mascara_fg = cv2.bitwise_and(mascara_rgb, mascara_rgb, mask=mascara_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, mascara_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y - traslacion_y:y - traslacion_y + h + movimientoY, x:x + w + movimientoX] = resultado

    # Mostrar el frame con la máscara aplicada
    cv2.imshow('Video con mascara', frame)

    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
video.release()
cv2.destroyAllWindows()

~~~

### Actividad procesamiento de color HSV en imagen
Para la actividad se usaron diferentes rango bajos y altos para los colores que se querian probar y se puso en un cv2.inRange() para hacer la mascara de cada color. Despues se uso un cv2.where() para aplicar la mascara y solo obtener los tonos del color deseado en la imagen.

~~~
import cv2
import numpy as np

imagen = cv2.imread('teoriaColor.jpg', 1)

imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

imagen_gris_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

#Imagen color verde
bajo_verde = np.array([40, 40, 40])
alto_verde = np.array([80, 255, 255])

mascara_verde = cv2.inRange(imagen_hsv, bajo_verde, alto_verde)

resultadoVerde = np.where(mascara_verde[:, :, None] == 255, imagen, imagen_gris_bgr)

#Imagen color Azul
bajo_azul = np.array([85, 40, 40])
alto_azul = np.array([128, 255, 255])

mascara_azul = cv2.inRange(imagen_hsv, bajo_azul, alto_azul)

resultadoAzul = np.where(mascara_azul[:, :, None] == 255, imagen, imagen_gris_bgr)

#Imagen color Amarillo
bajo_amarillo = np.array([22, 40, 40])
alto_amarillo = np.array([32, 255, 255])

mascara_amarillo = cv2.inRange(imagen_hsv, bajo_amarillo, alto_amarillo)

resultadoAmarillo = np.where(mascara_amarillo[:, :, None] == 255, imagen, imagen_gris_bgr)

#Imagen color Morado
bajo_morado = np.array([130, 40, 40])
alto_morado = np.array([145, 255, 255])

mascara_morado = cv2.inRange(imagen_hsv, bajo_morado, alto_morado)

resultadoMorado = np.where(mascara_morado[:, :, None] == 255, imagen, imagen_gris_bgr)


#Imagen color Rosado
bajo_rosa = np.array([146, 40, 40])
alto_rosa = np.array([162, 255, 255])

mascara_rosa = cv2.inRange(imagen_hsv, bajo_rosa, alto_rosa)

resultadoRosa = np.where(mascara_rosa[:, :, None] == 255, imagen, imagen_gris_bgr)

cv2.imshow('Color verde resaltado', resultadoVerde)
cv2.imshow('Color azul resaltado', resultadoAzul)
cv2.imshow('Color amarillo resaltado', resultadoAmarillo)
cv2.imshow('Color morado resaltado', resultadoMorado)
cv2.imshow('Color rosado resaltado', resultadoRosa)
cv2.imshow('Original', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

Para el video de igual forma se usa una mascara con rango de alto y bajo, en este caso para el color azul. En este caso la imagen es lo que se obtiene del video con cap.read() y se le aplica la mascara con un cv2.where() para obtener solo el color de la mascara azul.

~~~
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
        imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
~~~

### Actividad usar 5 mascaras con flujo optico
Para esta actividad replique 5 veces lo de la mascara de gas, y fui probando diferentes coordenadas y valores para poder acomodar lo mas posible cada imagen al rostro. Por ejemplo:
~~~
lentes_redimension = cv2.resize(mascara_lentes, (100, 60))
        nariz_redimension = cv2.resize(mascara_nariz, (100, 100))
        cuernos_redimension = cv2.resize(mascara_cuernos, (100, 100 ))
        sombrero_redimension = cv2.resize(mascara_sombrero, (w + 20, h + 10))
        bigote_redimension = cv2.resize(mascara_bigote, (100, 100))
~~~

cada imagen cuenta con un roi, que despues se asocia a un resultado, por ejemplo:
~~~
 # Crear una región de interés (ROI) en el frame donde colocaremos la máscara
        roi = frame[y:y + h + movimientoY, x:x + w + movimientoX]
        roi_lentes = frame[y + 30:y + 30 + 60, x + 20:x + 100 + 20]
        roi_nariz = frame[y + 45:y + 100 + 45, x + 20:x + 100 + 20]
        roi_cuernos = frame[y - 250:y + 100 - 250 , x + 20:x + 120]
        roi_sombrero = frame[y - 155:y + 10 + h - 155, x - 20:x + w + 20 - 20]
        roi_bigote = frame[y + 55:y + 100 + 55, x + 20:x + 100 + 20]

        if roi_lentes.shape[:2] == lentes_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            lentes_alpha_inv = cv2.bitwise_not(lentes_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi_lentes, roi_lentes, mask=lentes_alpha_inv)

            # Enmascarar la máscara RGB
            lentes_fg = cv2.bitwise_and(lentes_rgb, lentes_rgb, mask=lentes_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, lentes_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y + 30:y + 30 + 60, x + 20:x + 100+ 20] = resultado
~~~

Asi sucesivamente con cada una de las 5 las va a ir juntando con add con el fondo que seria la ventana, mostrando asi 5 mascaras.
condigo completo:
~~~
import cv2
import numpy as np

# Cargar la máscara que deseas agregar (asegúrate de que sea PNG con transparencia)
mascara_lentes = cv2.imread('lentes.png', cv2.IMREAD_UNCHANGED)  # Cargar PNG con transparencia
mascara_nariz = cv2.imread('nariz.png', cv2.IMREAD_UNCHANGED)
mascara_cuernos = cv2.imread('cuernos.png', cv2.IMREAD_UNCHANGED)
mascara_sombrero = cv2.imread('sombrero.png', cv2.IMREAD_UNCHANGED)
mascara_bigote = cv2.imread('bigote.png', cv2.IMREAD_UNCHANGED)

# Cargar el clasificador preentrenado de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Capturar video desde la cámara (o puedes usar un archivo de video)
video = cv2.VideoCapture(0)  # Cambia el 0 por la ruta de un archivo de video si quieres usar un archivo

while True:
    # Leer cada frame del video
    ret, frame = video.read()

    if not ret:
        break

    # Convertir el frame a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar los rostros en el frame
    rostros = face_cascade.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Procesar cada rostro detectado
    for (x, y, w, h) in rostros:
        movimientoX = 40
        movimientoY = 50 
        
        # Redimensionar la máscara para que coincida con el tamaño del rostro detectado
        lentes_redimension = cv2.resize(mascara_lentes, (100, 60))
        nariz_redimension = cv2.resize(mascara_nariz, (100, 100))
        cuernos_redimension = cv2.resize(mascara_cuernos, (100, 100 ))
        sombrero_redimension = cv2.resize(mascara_sombrero, (w + 20, h + 10))
        bigote_redimension = cv2.resize(mascara_bigote, (100, 100))
        
        lentes_rgb = lentes_redimension[:, :, :3] 
        lentes_alpha = lentes_redimension[:, :, 3] 
        
        nariz_rgb = nariz_redimension[:, :, :3] 
        nariz_alpha = nariz_redimension[:, :, 3] 
        
        cuernos_rgb = cuernos_redimension[:, :, :3] 
        cuernos_alpha = cuernos_redimension[:, :, 3] 
        
        sombrero_rgb = sombrero_redimension[:, :, :3] 
        sombrero_alpha = sombrero_redimension[:, :, 3] 
        
        bigote_rgb = bigote_redimension[:, :, :3] 
        bigote_alpha = bigote_redimension[:, :, 3] 
        
            
        if lentes_alpha.dtype != np.uint8:
            lentes_alpha = lentes_alpha.astype(np.uint8)
            
        if nariz_alpha.dtype != np.uint8:
            nariz_alpha = nariz_alpha.astype(np.uint8)
            
        if cuernos_alpha.dtype != np.uint8:
            cuernos_alpha = cuernos_alpha.astype(np.uint8)
            
        if sombrero_alpha.dtype != np.uint8:
            sombrero_alpha = sombrero_alpha.astype(np.uint8)
            
        if bigote_alpha.dtype != np.uint8:
            bigote_alpha = bigote_alpha.astype(np.uint8)

        traslacion_x = 20
        traslacion_y = 165

        # Crear una región de interés (ROI) en el frame donde colocaremos la máscara
        roi = frame[y:y + h + movimientoY, x:x + w + movimientoX]
        roi_lentes = frame[y + 30:y + 30 + 60, x + 20:x + 100 + 20]
        roi_nariz = frame[y + 45:y + 100 + 45, x + 20:x + 100 + 20]
        roi_cuernos = frame[y - 250:y + 100 - 250 , x + 20:x + 120]
        roi_sombrero = frame[y - 155:y + 10 + h - 155, x - 20:x + w + 20 - 20]
        roi_bigote = frame[y + 55:y + 100 + 55, x + 20:x + 100 + 20]
            
            
        if roi_lentes.shape[:2] == lentes_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            lentes_alpha_inv = cv2.bitwise_not(lentes_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi_lentes, roi_lentes, mask=lentes_alpha_inv)

            # Enmascarar la máscara RGB
            lentes_fg = cv2.bitwise_and(lentes_rgb, lentes_rgb, mask=lentes_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, lentes_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y + 30:y + 30 + 60, x + 20:x + 100+ 20] = resultado
            
        if roi_nariz.shape[:2] == nariz_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            nariz_alpha_inv = cv2.bitwise_not(nariz_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi_nariz, roi_nariz, mask=nariz_alpha_inv)

            # Enmascarar la máscara RGB
            nariz_fg = cv2.bitwise_and(nariz_rgb, nariz_rgb, mask=nariz_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, nariz_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y + 30:y + 100 + 30, x + 20:x + 100 + 20] = resultado
            
        if roi_cuernos.shape[:2] == cuernos_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            cuernos_alpha_inv = cv2.bitwise_not(cuernos_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi_cuernos, roi_cuernos, mask=cuernos_alpha_inv)

            # Enmascarar la máscara RGB
            cuernos_fg = cv2.bitwise_and(cuernos_rgb, cuernos_rgb, mask=cuernos_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, cuernos_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y - 250:y + 100 - 250 , x + 20:x + 120] = resultado
            
        if roi_sombrero.shape[:2] == sombrero_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            sombrero_alpha_inv = cv2.bitwise_not(sombrero_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi_sombrero, roi_sombrero, mask=sombrero_alpha_inv)

            # Enmascarar la máscara RGB
            sombrero_fg = cv2.bitwise_and(sombrero_rgb, sombrero_rgb, mask=sombrero_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, sombrero_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y - 155:y + 10 + h - 155, x - 20:x + w + 20 - 20] = resultado
            
        if roi_bigote.shape[:2] == bigote_rgb.shape[:2]:
            # Invertir la máscara alfa para obtener la parte del rostro donde se aplicará la máscara
            bigote_alpha_inv = cv2.bitwise_not(bigote_alpha)

            # Enmascarar la región del rostro en la imagen original
            fondo = cv2.bitwise_and(roi_bigote, roi_bigote, mask=bigote_alpha_inv)

            # Enmascarar la máscara RGB
            bigote_fg = cv2.bitwise_and(bigote_rgb, bigote_rgb, mask=bigote_alpha)

            # Combinar el fondo (parte del rostro sin máscara) y la parte con la máscara
            resultado = cv2.add(fondo, bigote_fg)

            # Reemplazar la región del rostro con la imagen combinada
            frame[y + 55:y + 100 + 55, x + 20:x + 100 + 20] = resultado

    # Mostrar el frame con la máscara aplicada
    cv2.imshow('Video con mascara', frame)

    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
video.release()
cv2.destroyAllWindows()

~~~

### Acitivdad Sistema solar

Para la actividad del sistema solar cada planeta tenia diferentes ejes, para simular la distancia de cada uno de los planetas (incluso el aro de saturno).
~~~
mercurio_ejeMayor = 200 
mercurio_ejeMenor = 150 

venus_ejeMayor = 230
venus_ejeMenor = 180

tierra_ejeMayor = 250
tierra_ejeMenor = 200

marte_ejeMayor = 270
marte_ejeMenor = 220

jupiter_ejeMayor = 310
jupiter_ejeMenor = 260

saturno_ejeMayor = 350
saturno_ejeMenor = 300
saturnoAro_ejeMayor = 355
saturnoAro_ejeMenor = 305

urano_ejeMayor = 380
urano_ejeMenor = 330

neptuno_ejeMayor = 400
neptuno_ejeMenor = 350
~~~

Tambien tenia cada uno un valor de movimiento diferente para simular el tiempo que tardan en girar alrededor del sol.
~~~
    t_mercurio = t * 1.6 
    t_venus = t * 1.4  
    t_tierra = t * 1.25  
    t_marte = t * 1.18  
    t_jupiter = t * 1 
    t_saturno = t * 0.9  
    t_saturnoAro = t * 0.9
    t_urano = t * 0.7  
    t_neptuno = t * 0.5 
~~~

Y cada planeta tiene diferente radio para simular su tamaño.
~~~
pt_tray_mercurio = generar_punto_elipse(mercurio_ejeMayor, mercurio_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_mercurio, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_Venus = generar_punto_elipse(venus_ejeMayor, venus_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_Venus, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_tierra = generar_punto_elipse(tierra_ejeMayor, tierra_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_tierra, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_marte = generar_punto_elipse(marte_ejeMayor, marte_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_marte, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_jupiter = generar_punto_elipse(jupiter_ejeMayor, jupiter_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_jupiter, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_saturno = generar_punto_elipse(saturno_ejeMayor, saturno_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_saturno, radius=1, color=(255, 255, 255), thickness=-1)
        pt_tray_saturnoAro = generar_punto_elipse(saturnoAro_ejeMayor, saturnoAro_ejeMenor, t_tray)
        
        pt_tray_urano = generar_punto_elipse(urano_ejeMayor, urano_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_urano, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_neptuno = generar_punto_elipse(neptuno_ejeMayor, neptuno_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_neptuno, radius=1, color=(255, 255, 255), thickness=-1)
    
~~~

__codig completo:__
~~~
import numpy as np
import cv2

def generar_punto_elipse(a, b, t):
    x = int(a * np.cos(t) + 500)  
    y = int(b * np.sin(t) + 500)
    return (x, y)

img_width, img_height = 1000, 1000

num_puntos = 1000

mercurio_ejeMayor = 200 
mercurio_ejeMenor = 150 

venus_ejeMayor = 230
venus_ejeMenor = 180

tierra_ejeMayor = 250
tierra_ejeMenor = 200

marte_ejeMayor = 270
marte_ejeMenor = 220

jupiter_ejeMayor = 310
jupiter_ejeMenor = 260

saturno_ejeMayor = 350
saturno_ejeMenor = 300
saturnoAro_ejeMayor = 355
saturnoAro_ejeMenor = 305

urano_ejeMayor = 380
urano_ejeMenor = 330

neptuno_ejeMayor = 400
neptuno_ejeMenor = 350

t_vals = np.linspace(0, 2 * np.pi, num_puntos)

for t in t_vals:

    imagen = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    t_mercurio = t * 1.6 
    t_venus = t * 1.4  
    t_tierra = t * 1.25  
    t_marte = t * 1.18  
    t_jupiter = t * 1 
    t_saturno = t * 0.9  
    t_saturnoAro = t * 0.9
    t_urano = t * 0.7  
    t_neptuno = t * 0.5 

    mercurioPunto = generar_punto_elipse(mercurio_ejeMayor, mercurio_ejeMenor, t_mercurio)
    veunsPunto = generar_punto_elipse(venus_ejeMayor, venus_ejeMenor, t_venus)
    tierraPunto = generar_punto_elipse(tierra_ejeMayor, tierra_ejeMenor, t_tierra)
    martePunto = generar_punto_elipse(marte_ejeMayor, marte_ejeMenor, t_marte)
    jupiterPunto = generar_punto_elipse(jupiter_ejeMayor, jupiter_ejeMenor, t_jupiter)
    saturnoPunto = generar_punto_elipse(saturno_ejeMayor, saturno_ejeMenor, t_saturno)
    saturnoAro = generar_punto_elipse(saturno_ejeMayor, saturno_ejeMenor, t_saturnoAro)
    uranoPunto = generar_punto_elipse(urano_ejeMayor, urano_ejeMenor, t_urano)
    neptunoPunto = generar_punto_elipse(neptuno_ejeMayor, neptuno_ejeMenor, t_neptuno)
    

    cv2.circle(imagen, mercurioPunto, radius=5, color=(126, 200, 243), thickness=-1)
    cv2.circle(imagen, veunsPunto, radius=8, color=(0, 100, 158), thickness=-1)
    cv2.circle(imagen, tierraPunto, radius=8, color=(76, 201, 166), thickness=-1)
    cv2.circle(imagen, martePunto, radius=7, color=(5, 80, 210), thickness=-1)
    cv2.circle(imagen, jupiterPunto, radius=11, color=(145, 218, 251), thickness=-1)
    cv2.circle(imagen, saturnoPunto, radius=11, color=(141, 188, 209), thickness=-1)
    cv2.circle(imagen, saturnoAro, radius=15, color=(89, 223, 255), thickness=1)
    cv2.circle(imagen, uranoPunto, radius=8, color=(243, 242, 169), thickness=-1)
    cv2.circle(imagen, neptunoPunto, radius=7, color=(221, 111, 60), thickness=-1)
    
    cv2.circle(imagen, (500, 500), 50, (0,206,255), -1)
    

    for t_tray in t_vals:
        pt_tray_mercurio = generar_punto_elipse(mercurio_ejeMayor, mercurio_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_mercurio, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_Venus = generar_punto_elipse(venus_ejeMayor, venus_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_Venus, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_tierra = generar_punto_elipse(tierra_ejeMayor, tierra_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_tierra, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_marte = generar_punto_elipse(marte_ejeMayor, marte_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_marte, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_jupiter = generar_punto_elipse(jupiter_ejeMayor, jupiter_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_jupiter, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_saturno = generar_punto_elipse(saturno_ejeMayor, saturno_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_saturno, radius=1, color=(255, 255, 255), thickness=-1)
        pt_tray_saturnoAro = generar_punto_elipse(saturnoAro_ejeMayor, saturnoAro_ejeMenor, t_tray)
        
        pt_tray_urano = generar_punto_elipse(urano_ejeMayor, urano_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_urano, radius=1, color=(255, 255, 255), thickness=-1)
        
        pt_tray_neptuno = generar_punto_elipse(neptuno_ejeMayor, neptuno_ejeMenor, t_tray)
        cv2.circle(imagen, pt_tray_neptuno, radius=1, color=(255, 255, 255), thickness=-1)
    

    cv2.imshow('img', imagen)
    

    cv2.waitKey(10)
    
cv2.destroyAllWindows()
~~~

### Filtro convolucional 1/9
Primero se creo una copia de la matriz original pero con bordes con 0.
~~~
matriz_con_borde = np.zeros((filas + 2, columnas + 2))
resultado = np.zeros((filas, columnas), dtype=np.uint8)

for i in range(filas):
    for j in range(columnas):
        matriz_con_borde[i + 1][j + 1] = imagen_gris[i][j]
~~~

Se aplica el filtro en ciclo for, pero usando los valores de la matriz con 0 para poder hacer la suma con los valores que estarian fuera de la matriz original, y el resultado de las sumas se guarda en la matriz original.
~~~


#Filtro 1/9 para la matriz
for i in range(filas):
    for j in range(columnas):
        resultado[i][j] = matriz_con_borde[x+i-1,y+j-1]*(1/9)+matriz_con_borde[x+i,y+j-1]*(1/9)+matriz_con_borde[x+i+1,y+j-1]*(1/9)+matriz_con_borde[x+i-1,y+j]*(1/9)+matriz_con_borde[x+i,y+j]*(1/9)+matriz_con_borde[x+i+1,y+j]*(1/9)+matriz_con_borde[x+i-1,y+j+1]*(1/9)+matriz_con_borde[x+i,y+j+1]*(1/9)+matriz_con_borde[x+i+1,y+j+1]*(1/9)
        

~~~

Dando como resultado una imagen un poco menos definida que la original
Original:
![Original](https://github.com/DavidMB4/Graficacion/blob/master/ActFiltroConvolucional/girasol.jpg?raw=true)


Filtro:
![Filtro](https://github.com/DavidMB4/Graficacion/blob/master/ActFiltroConvolucional/filtro%20.jpg?raw=true)

__Codigo completo:__
~~~
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
~~~

### Actividad pelota ping pong
Para que la esfera pueda moverse se le asigna velocidad y una posicion de inicio. La ventana se le asigna un acho y alto.
~~~
ancho, alto = 500, 500
imagen = 255 * np.ones((alto, ancho, 3), np.uint8)

inicioX, inicioY = 200, 200
velX, velY = 5, 7
~~~

Cada vez que la pelota llegue a una posicion mayor o igual al del borde su velocidad en X o en Y sera negativa para que vaya ahora al aldo contrario.
~~~
    inicioX = inicioX + velX
    inicioY = inicioY + velY
    
    if inicioX - 20 <= 0 or inicioX + 20 >= ancho:
        velX = -velX
    if inicioY - 20 <= 0 or inicioY + 20 >= alto:
        velY = -velY
~~~

### Proyecto 2 objetos con texturas
Para el proyecto 2 realizamos una granja, por mi parte hice un granero junto con unas vallas para que srivan como corrales
~~~
def draw_granero(madera_texture, madera_blanca_texture, techo_texture):
    draw_base(madera_texture)
    draw_base2(madera_texture)
    draw_base3(madera_texture)
    draw_techo(techo_texture)
    draw_techo2(techo_texture)
    draw_puerta(madera_blanca_texture, madera_texture)    
    draw_ventana_enfrente(madera_blanca_texture, madera_texture)
    draw_ventana(1, madera_blanca_texture)
    draw_ventana(2, madera_blanca_texture)
    draw_ventana(3, madera_blanca_texture)
    draw_ventana(4, madera_blanca_texture)
    draw_ventana(5, madera_blanca_texture)
    draw_ventana(6, madera_blanca_texture)


def draw_corral_cerdo(tierra_pasto_texture, madera_valla_texture, lodo_texture):
    positionsH = [
        (0, 0.3, 0),
        (0, 0.8, 0),
        (0, 1.3, 0),
        (0, 0.3, -8),
        (0, 0.8, -8),
        (0, 1.3, -8)
    ]
    for pos in positionsH:
        glPushMatrix()
        glTranslatef(*pos)
        valla_horizontal(madera_valla_texture)
        glPopMatrix()

    positionsV = [
        (0, 0.3, 0),
        (0, 0.8, 0),
        (0, 1.3, 0),
        (-8, 0.3, 0),
        (-8, 0.8, 0),
        (-8, 1.3, 0)
    ]
    for pos in positionsV:
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(90, 0.0, 1.0, 0.0)
        valla_horizontal(madera_valla_texture)
        glPopMatrix()
        
    positionP = [
        (3.8, 0,  0),
        (3.8, 0,  4),
        (3.8, 0,  -4),
        (-3.8, 0,  0),
        (-3.8, 0,  4),
        (-3.8, 0,  -4),
        (0, 0,  -4)
    ]
    for pos in positionP:
        glPushMatrix()
        glTranslatef(*pos)   
        poste_valla(madera_valla_texture)
        glPopMatrix()
        
    draw_tierra(tierra_pasto_texture)
    
    positionC = [
        (1, 0.03,  0),
        (1, 0.03,  1.5),
        (-0.5, 0.03,  0)
    ]
    for pos in positionC:
        glPushMatrix()
        glTranslatef(*pos)   
        draw_circle(0.0, 0.0, 2, 50, lodo_texture)
        glPopMatrix()
    

def draw_corral_oveja(tierra_pasto_texture, madera_valla_texture):
    positionsH = [
        (0, 0.3, 0),
        (0, 0.8, 0),
        (0, 1.3, 0),
        (0, 0.3, -8),
        (0, 0.8, -8),
        (0, 1.3, -8)
    ]
    for pos in positionsH:
        glPushMatrix()
        glTranslatef(*pos)
        valla_horizontal(madera_valla_texture)
        glPopMatrix()

    positionsV = [
        (0, 0.3, 0),
        (0, 0.8, 0),
        (0, 1.3, 0),
        (-8, 0.3, 0),
        (-8, 0.8, 0),
        (-8, 1.3, 0)
    ]
    for pos in positionsV:
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(90, 0.0, 1.0, 0.0)
        valla_horizontal(madera_valla_texture)
        glPopMatrix()
        
    positionP = [
        (3.8, 0,  0),
        (3.8, 0,  4),
        (3.8, 0,  -4),
        (-3.8, 0,  0),
        (-3.8, 0,  4),
        (-3.8, 0,  -4),
        (0, 0,  -4)
    ]
    for pos in positionP:
        glPushMatrix()
        glTranslatef(*pos)   
        poste_valla(madera_valla_texture)
        glPopMatrix()
        
    draw_tierra(tierra_pasto_texture)
~~~

Para ponerle las texturas descargue imagenes en jpg, puse la funcion load_texture()
~~~
def load_texture(texture_file):
    img = Image.open(texture_file)
    img_data = img.tobytes("raw", "RGB", 0, -1)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    return texture_id
~~~
A esta funcion se le pasa como atributo la ruta de las imagenes y se guarda en una varible cada una.
~~~
    madera_texture = load_texture('madera.jpg')
    madera_blanca_texture = load_texture('madera_blanca.jpg')
    techo_texture = load_texture('techo.jpg')
    tierra_pasto_texture = load_texture('tierra_pasto.jpg')
    madera_valla_texture = load_texture('madera_valla.jpg')
    lodo_texture = load_texture('lodo.jpg')
~~~

En el init se habilitan las texturas con: glEnable(GL_TEXTURE_2D) y se usa la funcion glBindTexture(GL_TEXTURE_2D, texture) para vincular la textura al glBindTexture(GL_TEXTURE_2D, texture), esto lo activa y se tiene que poner al inicio de las funciones que crean las partes de los objetos por ejemplo las vallas. Tambien se usa glTexCoord2f(); para establecer las coordenadas de las texturas en cada funcion de los vertices. Puede decirse que es como de donde a donde va a ir la textura (se estira o encoge para cubrir la posicion de los vertices). Un ejemplo seria en las vallas:
~~~
def valla_horizontal(texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    glColor3f(1, 1, 1)  
        # Frente
    glTexCoord2f(0.0, 0.0); glVertex3f(-4, 0.3, 4)
    glTexCoord2f(1.0, 0.0); glVertex3f(4, 0.3, 4)
    glTexCoord2f(1.0, 1.0); glVertex3f(4, 0.7, 4)
    glTexCoord2f(0.0, 1.0); glVertex3f(-4, 0.7, 4)
        
    glEnd()
~~~

Como se ve cada objeto con texturas:
![Granero y corrales con texturas](https://github.com/DavidMB4/Graficacion/blob/master/Actividad1Transformaciones/pokeball.png?raw=true)