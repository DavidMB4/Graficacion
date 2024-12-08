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
