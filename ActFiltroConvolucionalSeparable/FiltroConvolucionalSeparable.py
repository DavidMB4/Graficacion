import numpy as np

matriz = [
    [12,12,13,14,180],
    [16,16,13,120,18],
    [70,13,13,12,17],
    [16,17,18,18,17],
    [120,130,130,12,160]
]

filtro = [
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

filas, columnas = len(matriz), len(matriz[0])

y=1
x=1

matriz_con_borde = np.zeros((filas + 2, columnas + 2))

resultado = np.zeros((filas, columnas), dtype=int)

for i in range(filas):
    for j in range(columnas):
        matriz_con_borde[i + 1][j + 1] = matriz[i][j]

#Matriz con el filtro completo
for i in range(filas):
    for j in range(columnas):
        resultado[i][j] = matriz_con_borde[x+i-1,y+j-1]*(-1)+matriz_con_borde[x+i,y+j-1]*(-2)+matriz_con_borde[x+i+1,y+j-1]*(-1)+matriz_con_borde[x+i-1,y+j]*(0)+matriz_con_borde[x+i,y+j]*(0)+matriz_con_borde[x+i+1,y+j]*(0)+matriz_con_borde[x+i-1,y+j+1]*(1)+matriz_con_borde[x+i,y+j+1]*(2)+matriz_con_borde[x+i+1,y+j+1]*(1)
        print(resultado[i][j])


resultado[i][j] = int(resultado[i][j])
print("Matriz después de aplicar el filtro convolucional:")
for fila in resultado:
    print(fila)
    
    
#Matriz con filtro solo con la fila y la columna
for i in range(filas):
    for j in range(columnas):
        resultado[i][j] = matriz_con_borde[x+i-1,y+j-1]*(1)+matriz_con_borde[x+i,y+j-1]*(2)+matriz_con_borde[x+i+1,y+j-1]*(1)+matriz_con_borde[x+i-1,y+j+1]*(1)+matriz_con_borde[x+i,y+j+1]*(2)+matriz_con_borde[x+i+1,y+j+1]*(1)
        print(resultado[i][j])


resultado[i][j] = int(resultado[i][j])
print("Matriz después de aplicar el filtro:")
for fila in resultado:
    print(fila)