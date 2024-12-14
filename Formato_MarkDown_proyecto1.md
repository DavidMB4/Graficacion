# Proyecto 1: Rotacion, Escalamiento, y Traslacion con flujo optico
Para el proyecto 1 use una imagen png y los codigos que habiamos visto en clase de la malla y de la mascara de gas.
El codigo tiene el contador de lo que sera el valor de la traslacion, escalamiento y rotacion, y los limites de cada uno:
~~~
limTrasPos = 100  # Máximo de traslación a la derecha
limTrasNeg = -100  # Puedes usar este para traslación negativa si es necesario
conTras = 0  # Contador de la traslación

limEscMax = 3.0  # Máximo factor de escalamiento
limEscMin = 0.6  # Mínimo factor de escalamiento
conEsc = 1  # Factor de escalamiento inicial

limRotHoraria = 360
limRotAntiHorario = -360
conRot = 0
~~~

Use el cv.flip para que se reflejara la imagen que obtiene el frame y poder manejar mejor el flujo optico
~~~
frame = cv.flip(frame, 1)
~~~

Calcula el tamaño para el escalamiento, teniendo un tamaño original de 100 el cual puede ir aumentando con conEsc y lo convierte en entero para poder usarse en cv.resize
~~~
tamano = int(100 * conEsc)  

    # Redimensiona la imagen PNG según el tamaño
    resized_png = cv.resize(png_image, (tamano, tamano))
~~~

Otiene la matriz de rotacion apartir de obtener la X y Y de la imagen, lo cual sirve para obtener el centro de rotacion de la imagen y usar cv.getRotationMatrix2D para crear una matriz de rotacion, donde usaremos conRot que servira como el angulo, siendo un angulo positivo lo que hara que gire en sentido antihorario y un negativo que lo haga en sentido horario. Despues se usa un warpAffine para aplicar la rotacion a la imagen con la matriz creada. 
~~~
    x, y = resized_png.shape[:2]
    centro_imagen = (y // 2, x // 2)
    Matriz_rotacion = cv.getRotationMatrix2D(centro_imagen, conRot, 1.0)
    
    imagen_rotada = cv.warpAffine(resized_png, Matriz_rotacion, (y,x))
~~~

Tambien se usan los canales RGB y alpha de la imagen ya que es un png y se tienen que usar tambien.
~~~
    mascara_rgb = imagen_rotada[:, :, :3]
    mascara_alpha = imagen_rotada[:, :, 3]
~~~

Se usa un roi que servira la region donde se pondra la imagen y de esta forma podemos mover la imagen con el conTras.
~~~
# Determinar la posición central
    x_centro = (ancho - tamano) // 2
    y_centro = (alto - tamano) // 2

    # Crear una región de interés (ROI) en el centro del frame
    roi = frame[y_centro:y_centro + tamano, x_centro + conTras:x_centro + conTras + tamano]
~~~
Ya que de esta forma no solo ponemos la imagen en el centro del frame si no tambien la vamos moviendo con los valores de la traslacion.

Se combinan los valores de las mascaras de la imagen (RGB, alpha) y se juntando con el fondo que seria el frame de esta forma se combina la imagen la imagen del frame.
~~~
# Combinar la máscara con el ROI del frame
    mascara_alpha_inv = cv.bitwise_not(mascara_alpha)
    fondo = cv.bitwise_and(roi, roi, mask=mascara_alpha_inv)
    mascara_fg = cv.bitwise_and(mascara_rgb, mascara_rgb, mask=mascara_alpha)
    resultado = cv.add(fondo, mascara_fg)

    # Colocar la imagen combinada en el centro del frame
    frame[y_centro:y_centro + tamano, x_centro + conTras:x_centro + conTras + tamano] = resultado
~~~

Despues hace el calculo que hace en la malla, donde hace un arreglo con las opsiciones de los puntos que va a hacer. Tambien se inicializan en 0 unas variables que serviran para hacer que se traslade, rote o escale la imagen, y reducir que lo haga por imprevistos como el cambio de luz o que pase algo detras de la imagen 
~~~
if p1 is None:
        vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)
        p0 = np.array([(50, 400), (50, 420), (600, 400), (600, 420), (250, 100), (250, 120), (290, 110), (60, 410), (590, 410)])
        p0 = np.float32(p0[:, np.newaxis, :])
        mask = np.zeros_like(vframe)
        cv.imshow('ventana', frame)
    else:
        bp1 = p1[st == 1]
        bp0 = p0[st == 1]
        
        T0x, T0y, T1x, T1y, R4x, R4y, E2x, E2y, E3x, E3y, R5x, R5y, R6x, R6y, E8x, E8y, T7x, T7y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
~~~

Cada punto tiene 2 circulos y una linea que los une, el circle orignal tiene X y Y como c y d, y el circle nuevo que aparece en la posicion donde detecta el movimiento tiene X y Y como a y b, y la linea los une.
~~~
frame = cv.line(frame, (c, d), (a, b), (0, 0, 255), 2)
            frame = cv.circle(frame, (c, d), 2, (255, 0, 0), -1)
            frame = cv.circle(frame, (a, b), 3, (0, 255, 0), -1)

            distancia = math.sqrt((c - a) ** 2 + (d - b) ** 2)
            if i == 0: 
                T0x = c - a
                T0y = d - b
                    
            if i == 1:  
                T1x = c -a
                T1y = d -b
                
            if i == 7:
                T7x = c - a
                T7y = d - b
                
            if i == 8:
                E8x = c - a
                E8y = d - b
                    
            if i == 2:
                E2x = c - a
                E2y = d - b
                            
            if i == 3:
                E3x = c -a
                E3y = d - b
                
            if i == 4:
                R4x = c - a
                R4y = d - b
                
            if i == 5:
                R5x = c - a
                R5y = d - b
            if i == 6:
                R6x = c - a
                R6y = d - b
~~~
En un primer inicia se calculaba la magnitud de la distancia entre el circcle original y el de la posicion nueva para validar si se traslada, rotaba o escalaba, pero no funionaba bien a la hora de evitar que algo externo afectara a la imagen, como la luz. 
Al final use el manejo de las coordenadas de los circle, ya que esto ayuda a comprobar si esta moviendo a la derecha o izquierda en X, y abajo o arriba en Y. Se e resta el valor nuevo de la posicion al valor original, y se guarda en cada una de las varibales, siempre y cuando el punto que detecte movimiento sea el mismo que el valor de i dentro del arrelgo de las posiciones.

Ahora por ejemplo para comprobar que es el usuario el que esta moviendo los puntos, el usuario tendria que mover la mano de tal forma que el valor absoluto de las variable que empiezan con T (traslacion) sean mayores a 10 en el caso de los valores de X y mayores a 5 en el caso de Y, esto quiere decir que se estan moviendo todos los puntos, ahora para reducir mas el que se mueva de forma accidental, se usan los valores nromales de T, por ejemplo sii T0x fuera mayor que 0 significa el cicle nuevo en x se movio a la izqiuera, por lo tanto x en el cicrcle original es mayor asi que, al restar c - a daria un resultado positivo, esto indica que el movimiento lo reconocio hacia la izquierda de las x. Ahora con T0y fuera menor que 0 significa que el circle nuevo se movio hacia abajo en y que en este caso significa que es un valor mayor de y que el valor original de circle. Por lo tanto al restar d - b daria un resultado negativo, lo que indica que se reconocio el moviento hacia abajo en Y. 

~~~
 if abs(T0x) > 10 and abs(T1x) > 10 and abs(T0y) > 5 and abs(T1y) > 5 and abs(T7x) > 10 and abs(T7y) > 5:
            if T0x > 0 and T1x > 0 and T0y < 0 and T1y < 0 and T7x > 0 and T7y < 0:
~~~
Si esto es cierto entonces conTras disminuya su valor en 5. Esto significa que en los calculos del roi, en el siguiente ciclo la iamgen aparecera un poco hacia la izquierda. 
~~~
 if abs(T0x) > 10 and abs(T1x) > 10 and abs(T0y) > 5 and abs(T1y) > 5 and abs(T7x) > 10 and abs(T7y) > 5:
            if T0x > 0 and T1x > 0 and T0y < 0 and T1y < 0 and T7x > 0 and T7y < 0:
                if conTras > limTrasNeg: 
                    conTras -= 5 # si los puntos van a la izquierda en diagonal hacia abajo se resta a la trasclacion
                    print(conTras)
~~~

En otras palabras los puntos de la izquierda deben moverse hacia la izquierda y un poco hacia abajo (en diagonal hacia abajo) para que la imagen haga una traslacion poco a poco hacia la izquierda de las X.

En el codigo se hace algo parecido para cada una de las actividades, el tema es mover la imagen en los puntos hacia la izquierda en diagonal hacia abajo o hacia la derecha en diagonal hacia abajo. 

__codigo completo:__
~~~
import numpy as np
import cv2 as cv
import math

cap = cv.VideoCapture(0)

lkparm = dict(winSize=(15, 15), maxLevel=2,
              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

_, vframe = cap.read()
vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)
p0 = np.array([(50, 400), (50, 420), (600, 400), (600, 420), (250, 100), (250, 120), (290, 110), (60, 410), (590, 410)])

p0 = np.float32(p0[:, np.newaxis, :])

mask = np.zeros_like(vframe)
cad = ''

limTrasPos = 100  # Máximo de traslación a la derecha
limTrasNeg = -100  # Puedes usar este para traslación negativa si es necesario
conTras = 0  # Contador de la traslación

limEscMax = 3.0  # Máximo factor de escalamiento
limEscMin = 0.6  # Mínimo factor de escalamiento
conEsc = 1  # Factor de escalamiento inicial

limRotHoraria = 360
limRotAntiHorario = -360
conRot = 0

# Carga la imagen PNG con canal alfa
png_image = cv.imread("flor.png", cv.IMREAD_UNCHANGED)


while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    fgris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(vgris, fgris, p0, None, **lkparm)

    # Dimensiones del marco
    alto, ancho, _ = frame.shape
    centro_x, centro_y = ancho // 2, alto // 2
    tamano = int(100 * conEsc)  

    # Redimensiona la imagen PNG según el tamaño
    resized_png = cv.resize(png_image, (tamano, tamano))
    
    x, y = resized_png.shape[:2]
    centro_imagen = (y // 2, x // 2)
    Matriz_rotacion = cv.getRotationMatrix2D(centro_imagen, conRot, 1.0)
    
    imagen_rotada = cv.warpAffine(resized_png, Matriz_rotacion, (y,x))

    # Separar los canales de la máscara
    mascara_rgb = imagen_rotada[:, :, :3]
    mascara_alpha = imagen_rotada[:, :, 3]
    
    # Determinar la posición central
    x_centro = (ancho - tamano) // 2
    y_centro = (alto - tamano) // 2

    # Crear una región de interés (ROI) en el centro del frame
    roi = frame[y_centro:y_centro + tamano, x_centro + conTras:x_centro + conTras + tamano]
    
    # Combinar la máscara con el ROI del frame
    mascara_alpha_inv = cv.bitwise_not(mascara_alpha)
    fondo = cv.bitwise_and(roi, roi, mask=mascara_alpha_inv)
    mascara_fg = cv.bitwise_and(mascara_rgb, mascara_rgb, mask=mascara_alpha)
    resultado = cv.add(fondo, mascara_fg)

    # Colocar la imagen combinada en el centro del frame
    frame[y_centro:y_centro + tamano, x_centro + conTras:x_centro + conTras + tamano] = resultado

    if p1 is None:
        vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)
        p0 = np.array([(50, 400), (50, 420), (600, 400), (600, 420), (250, 100), (250, 120), (290, 110), (60, 410), (590, 410)])
        p0 = np.float32(p0[:, np.newaxis, :])
        mask = np.zeros_like(vframe)
        cv.imshow('ventana', frame)
    else:
        bp1 = p1[st == 1]
        bp0 = p0[st == 1]
        
        T0x, T0y, T1x, T1y, R4x, R4y, E2x, E2y, E3x, E3y, R5x, R5y, R6x, R6y, E8x, E8y, T7x, T7y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for i, (nv, vj) in enumerate(zip(bp1, bp0)):
            a, b = (int(x) for x in nv.ravel())
            c, d = (int(x) for x in vj.ravel())
            

            frame = cv.line(frame, (c, d), (a, b), (0, 0, 255), 2)
            frame = cv.circle(frame, (c, d), 2, (255, 0, 0), -1)
            frame = cv.circle(frame, (a, b), 3, (0, 255, 0), -1)

            distancia = math.sqrt((c - a) ** 2 + (d - b) ** 2)
            if i == 0: 
                T0x = c - a
                T0y = d - b
                    
            if i == 1:  
                T1x = c -a
                T1y = d -b
                
            if i == 7:
                T7x = c - a
                T7y = d - b
                
            if i == 8:
                E8x = c - a
                E8y = d - b
                    
            if i == 2:
                E2x = c - a
                E2y = d - b
                            
            if i == 3:
                E3x = c -a
                E3y = d - b
                
            if i == 4:
                R4x = c - a
                R4y = d - b
                
            if i == 5:
                R5x = c - a
                R5y = d - b
            if i == 6:
                R6x = c - a
                R6y = d - b
                
        if abs(T0x) > 10 and abs(T1x) > 10 and abs(T0y) > 5 and abs(T1y) > 5 and abs(T7x) > 10 and abs(T7y) > 5:
            if T0x > 0 and T1x > 0 and T0y < 0 and T1y < 0 and T7x > 0 and T7y < 0:
                if conTras > limTrasNeg: 
                    conTras -= 5 # si los puntos van a la izquierda en diagonal hacia abajo se resta a la trasclacion
                    print(conTras)
            elif T0x < 0 and T1x < 0 and T0y < 0 and T1y < 0 and T7x < 0 and T7y < 0:
                if conTras < limTrasPos:
                    conTras += 5 # si los puntos van a la derecha en diagonal hacia abajo se suma a la trasclacion
                    print(conTras)
                    
        if abs(E2x) > 10 and abs(E3x) > 10 and abs(E2y) > 5 and abs(E3y) > 5 and abs(E8x) > 5 and abs(E8y) > 5: 
            if E2x > 0 and E3x > 0 and E2y < 0 and E3y < 0 and E8x > 0 and E8y < 0:
                if conEsc < limEscMax:
                     conEsc += 0.2 # si los puntos van a la izquierda en diagonal hacia abajo se aumenta el escalamiento
                     print(conEsc)
            elif E2x < 0 and E2x < 0 and E2y < 0 and E3y < 0 and E8x < 0 and E8y < 0:
                if limEscMin < conEsc:
                    conEsc -= 0.2 # si los puntos van a la derecha en diagonal hacia abajo se reduce el escalamiento 
                    print(conEsc)
                    
        if abs(R4x) > 10 and abs(R5x) > 10 and abs(R6x) > 10 and abs(R4y) > 5 and abs(R5y) > 5 and abs(R6y) > 5:
            if R4x > 0 and R5x > 0 and R6x > 0 and R4y < 0 and R5y < 0 and R6y < 0:
                if conRot < limRotHoraria:
                    conRot += 5  # si los puntos van a la derecha en diagonal hacia abajo rota en sentido antihorario
                    print("rotacion antihoraria")
            elif R4x < 0 and R5x < 0 and R6x < 0 and R4y < 0 and R5y < 0 and R6y < 0:
                if limRotAntiHorario < conRot:
                    conRot -= 5 # si los puntos van a la izquierda en diagonal hacia abajo rota en sentido horario
                    print("rotacion horaria")
                
                    


        cv.imshow('ventana', frame)

        vgris = fgris.copy()

    if (cv.waitKey(1) & 0xff) == 27:  # Salir con 'Esc'
        break

cap.release()
cv.destroyAllWindows()
~~~