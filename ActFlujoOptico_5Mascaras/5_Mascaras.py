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
