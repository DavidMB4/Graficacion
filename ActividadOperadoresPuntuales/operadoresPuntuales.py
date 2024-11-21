import cv2 as cv

img = cv.imread(r'C:\Users\david\OneDrive\Documentos\Tec\5 Semestre\Graficacion\Repositorio\ActividadOperadoresPuntuales\imagen.jpg', 0)

img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
img5 = img.copy()
img6 = img.copy()

x,y=img.shape
#Primer operador
for i in range(x):
        for j in range(y):
                if(img2[i,j]>150):
                        img2[i,j]=200
                else:
                        img2[i,j]=10

#Segundo operador
for i in range(x):
        for j in range(y):
                if(img3[i,j]==150 or img3[i,j]==200):
                        img3[i,j]=255
                else:
                        img3[i,j]=75

#Tercer operador
for i in range(x):
        for j in range(y):
                if(img4[i,j]>=100 and img4[i,j]<=200):
                        img4[i,j]=45

#Cuarto operador
for i in range(x):
        for j in range(y):
                if(img5[i,j]<150):
                        img5[i,j]=255
                else:
                        img5[i,j]=100

#Quinto operador
for i in range(x):
        for j in range(y):
                if(img6[i,j]==255):
                        img6[i,j]=0



cv.imshow('original', img)
cv.imshow('Primer operador', img2)
cv.imshow('Segundo operador', img3)
cv.imshow('Tercer operador', img4)
cv.imshow('Cuardo operador', img5)
cv.imshow('Quinto operador', img6)

print( img.shape, x , y)
cv.waitKey(0)
cv.destroyAllWindows()