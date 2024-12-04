import numpy as np
import cv2 as cv
import math

cap = cv.VideoCapture(0)

lkparm = dict(winSize=(15, 15), maxLevel=2,
              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

_, vframe = cap.read()
vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)
p0 = np.array([(50, 400), (600, 400), (50, 100), (300, 100), (600, 100)])

p0 = np.float32(p0[:, np.newaxis, :])

mask = np.zeros_like(vframe)
cad = ''

limTrasPos = 100  # Máximo de traslación a la derecha
limTrasNeg = -100  # Puedes usar este para traslación negativa si es necesario
conTras = 0  # Contador de la traslación

limEscMax = 3.0  # Máximo factor de escalamiento
limEscMin = 0.6  # Mínimo factor de escalamiento
conEsc = 1  # Factor de escalamiento inicial

conRot = 0

# Carga la imagen PNG con canal alfa
png_image = cv.imread("flor.png", cv.IMREAD_UNCHANGED)

# Antes de entrar al bucle, inicializamos la matriz de transformación para la traslación
M_translation = np.float32([[1, 0, conTras], [0, 1, 0]])

while True:
    _, frame = cap.read()
    fgris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(vgris, fgris, p0, None, **lkparm)

    # Dimensiones del marco
    alto, ancho, _ = frame.shape
    centro_x, centro_y = ancho // 2, alto // 2
    tamano = int(100 * conEsc) # max(int(100 * conEsc), 1)   # Tamaño del lado del cuadrado

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
        p0 = np.array([(50, 400), (600, 400), (50, 100), (300, 100), (600, 100)])
        p0 = np.float32(p0[:, np.newaxis, :])
        mask = np.zeros_like(vframe)
        cv.imshow('ventana', frame)
    else:
        bp1 = p1[st == 1]
        bp0 = p0[st == 1]

        for i, (nv, vj) in enumerate(zip(bp1, bp0)):
            a, b = (int(x) for x in nv.ravel())
            c, d = (int(x) for x in vj.ravel())

            frame = cv.line(frame, (c, d), (a, b), (0, 0, 255), 2)
            frame = cv.circle(frame, (c, d), 2, (255, 0, 0), -1)
            frame = cv.circle(frame, (a, b), 3, (0, 255, 0), -1)

            distancia = math.sqrt((c - a) ** 2 + (d - b) ** 2)
            if i == 0 and distancia > 50:  # Si la distancia es mayor a 11
                print("Derecha")
                if conTras < limTrasPos:  # Si no hemos alcanzado el límite de desplazamiento
                    conTras += 5  # Mover la figura 5 píxeles hacia la derecha
                    
            if i == 2 and distancia > 50:  # Si la distancia es mayor a 11
                print("Izquierda")
                if conTras > limTrasNeg:  # Si no hemos alcanzado el límite de desplazamiento -
                    conTras -= 5  # Mover la figura 5 píxeles hacia la izquierda
                    
            if i == 1 and distancia > 50:  # Punto 2 para escalamiento
                print("Aumenta escala")
                if conEsc < limEscMax:
                    conEsc += 0.2  # Incrementa el escalamiento
                    print(ancho)
                            
            if i == 4 and distancia > 50:  # Detectar movimiento en el punto 5 si la distancia es mayor a 11
                print("reduce escala")
                if limEscMin < conEsc:
                    conEsc -= 0.2  # Reducir el factor de escala en 0.2
                
            if i == 3 and distancia > 50:  # Punto 2 para rotación
                conRot += 5  # Incrementa la rotación en 5 grados
                    


        cv.imshow('ventana', frame)

        vgris = fgris.copy()

    if (cv.waitKey(1) & 0xff) == 27:  # Salir con 'Esc'
        break

cap.release()
cv.destroyAllWindows()
