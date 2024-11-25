import numpy as np 
import cv2 as cv
import math

piramide = np.zeros((6, 11))

for i in range(6):
    for j in range(11):
        if i == 0:
            piramide[0][5] = 1    
        else:
            if j == 10 :
                piramide[i][j] = piramide[i-1][j-1]
            
            elif j == 0: 
                piramide[i][j] = piramide[i-1][j+1]
            
            else:
                piramide[i][j] = piramide[i-1][j-1] + piramide[i-1][j+1]
            
print(piramide)