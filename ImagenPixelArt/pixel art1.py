import numpy as np
import cv2

image = np.ones((600, 600, 3), dtype=np.uint8) * 255

x, y = 275, 255
l=5

#parte negra
#cabeza
cv2.rectangle(image, (x+2*l, y), (x+14*l, y+1*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+17*l, y), (x+19*l, y+1*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+14*l, y+1*l), (x+17*l, y+2*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+1*l, y+1*l), (x+2*l, y+2*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+2*l, y+2*l), (x+3*l, y+3*l), (0, 0, 0), -1)     
cv2.rectangle(image, (x+3*l, y+3*l), (x+4*l, y+4*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+4*l, y+4*l), (x+5*l, y+5*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+3*l, y+5*l), (x+4*l, y+6*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+2*l, y+6*l), (x+3*l, y+7*l), (0, 0, 0), -1)     
cv2.rectangle(image, (x+1*l, y+7*l), (x+2*l, y+8*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x, y+8*l), (x+1*l, y+9*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x, y+8*l), (x+6*l, y+9*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+4*l, y+9*l), (x+5*l, y+10*l), (0, 0, 0), -1)     
cv2.rectangle(image, (x+3*l, y+10*l), (x+4*l, y+11*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+2*l, y+11*l), (x+3*l, y+12*l), (0, 0, 0), -1)     
cv2.rectangle(image, (x+1*l, y+12*l), (x+2*l, y+13*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+1, y+13*l), (x+10*l, y+14*l), (0, 0, 0), -1)   

cv2.rectangle(image, (x+18*l, y), (x+19*l, y+4*l), (0, 0, 0), -1)       
cv2.rectangle(image, (x+19*l, y+4*l), (x+20*l, y+7*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+18*l, y+7*l), (x+19*l, y+13*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+17*l, y+9*l), (x+20*l, y+10*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+14*l, y+9*l), (x+15*l, y+6*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+10*l, y+14*l), (x+17*l, y+15*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+17*l, y+13*l), (x+18*l, y+14*l), (0, 0, 0), -1)

cv2.rectangle(image, (x+16*l, y+16*l), (x+18*l, y+15*l), (0, 0, 0), -1) 


#brazos
cv2.rectangle(image, (x+8*l, y+14*l), (x+9*l, y+15*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+7*l, y+15*l), (x+8*l, y+16*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+9*l, y+15*l), (x+11*l, y+16*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+6*l, y+16*l), (x+7*l, y+17*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+8*l, y+16*l), (x+9*l, y+17*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+6*l, y+17*l), (x+8*l, y+18*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+8*l, y+17*l), (x+9*l, y+19*l), (0, 0, 0), -1)    
cv2.rectangle(image, (x+9*l, y+19*l), (x+10*l, y+20*l), (0, 0, 0), -1)  
cv2.rectangle(image, (x+10*l, y+20*l), (x+12*l, y+22*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+12*l, y+19*l), (x+13*l, y+20*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+13*l, y+19*l), (x+14*l, y+16*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+11*l, y+16*l), (x+13*l, y+15*l), (0, 0, 0), -1) 

cv2.rectangle(image, (x+16*l, y+21*l), (x+17*l, y+16*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+16*l, y+17*l), (x+20*l, y+16*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+19*l, y+16*l), (x+20*l, y+19*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+18*l, y+19*l), (x+19*l, y+21*l), (0, 0, 0), -1) 


#pies
cv2.rectangle(image, (x+11*l, y+22*l), (x+9*l, y+23*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+9*l, y+23*l), (x+10*l, y+27*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+9*l, y+27*l), (x+21*l, y+28*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+20*l, y+27*l), (x+21*l, y+26*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+20*l, y+26*l), (x+19*l, y+25*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+19*l, y+25*l), (x+17*l, y+24*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+17*l, y+24*l), (x+15*l, y+23*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+15*l, y+23*l), (x+13*l, y+22*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+14*l, y+21*l), (x+18*l, y+20*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+14*l, y+22*l), (x+15*l, y+21*l), (0, 0, 0), -1) 

cv2.rectangle(image, (x+16*l, y+27*l), (x+17*l, y+26*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+16*l, y+26*l), (x+14*l, y+25*l), (0, 0, 0), -1) 
cv2.rectangle(image, (x+14*l, y+25*l), (x+12*l, y+24*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+13*l, y+24*l), (x+12*l, y+22*l), (0, 0, 0), -1)

    

#cabeza color
cv2.rectangle(image, (x+2*l, y+1*l), (x+14*l, y+2*l), (189,112,2), -1) 
cv2.rectangle(image, (x+17*l, y+1*l), (x+18*l, y+2*l), (189,112,2), -1) 
cv2.rectangle(image, (x+3*l, y+2*l), (x+6*l, y+3*l), (189,112,2), -1) 
cv2.rectangle(image, (x+6*l, y+2*l), (x+18*l, y+3*l), (189,112,2), -1) 
cv2.rectangle(image, (x+4*l, y+3*l), (x+6*l, y+4*l), (189,112,2), -1) 
cv2.rectangle(image, (x+6*l, y+3*l), (x+18*l, y+4*l), (189,112,2), -1) 
cv2.rectangle(image, (x+5*l, y+4*l), (x+6*l, y+5*l), (189,112,2), -1) 
cv2.rectangle(image, (x+6*l, y+4*l), (x+11*l, y+5*l), (189,112,2), -1) 
cv2.rectangle(image, (x+15*l, y+4*l), (x+19*l, y+5*l), (189,112,2), -1) 

cv2.rectangle(image, (x+15*l, y+5*l), (x+19*l, y+6*l), (189,112,2), -1) 
cv2.rectangle(image, (x+4*l, y+5*l), (x+10*l, y+6*l), (189,112,2), -1) 
cv2.rectangle(image, (x+3*l, y+6*l), (x+10*l, y+7*l), (189,112,2), -1) 
cv2.rectangle(image, (x+15*l, y+6*l), (x+19*l, y+7*l), (189,112,2), -1) 
cv2.rectangle(image, (x+2*l, y+7*l), (x+10*l, y+8*l), (189,112,2), -1) 
cv2.rectangle(image, (x+15*l, y+7*l), (x+17*l, y+8*l), (189,112,2), -1)
cv2.rectangle(image, (x+6*l, y+8*l), (x+10*l, y+9*l), (189,112,2), -1) 
cv2.rectangle(image, (x+16*l, y+8*l), (x+17*l, y+9*l), (189,112,2), -1) 
cv2.rectangle(image, (x+5*l, y+9*l), (x+11*l, y+10*l), (189,112,2), -1) 
cv2.rectangle(image, (x+4*l, y+10*l), (x+8*l, y+11*l), (189,112,2), -1) 
cv2.rectangle(image, (x+8*l, y+10*l), (x+11*l, y+11*l), (142,191,251), -1)
cv2.rectangle(image, (x+15*l, y+10*l), (x+18*l, y+11*l), (142,191,251), -1)
cv2.rectangle(image, (x+3*l, y+11*l), (x+8*l, y+12*l), (189,112,2), -1) 
cv2.rectangle(image, (x+8*l, y+11*l), (x+18*l, y+12*l), (142,191,251), -1)
cv2.rectangle(image, (x+2*l, y+12*l), (x+8*l, y+13*l), (189,112,2), -1) 
cv2.rectangle(image, (x+8*l, y+12*l), (x+18*l, y+13*l), (142,191,251), -1)
cv2.rectangle(image, (x+10*l, y+13*l), (x+17*l, y+14*l), (142,191,251), -1)
cv2.rectangle(image, (x+17*l, y+13*l), (x+16*l, y+14*l), (0,0,0), -1)
    
    
#oreja
cv2.rectangle(image, (x+6*l, y+5*l), (x+7*l, y+2*l), (0, 0, 0), -1) 
#oreja color
cv2.rectangle(image, (x+7*l, y+5*l), (x+8*l, y+2*l), (142,191,251), -1) 
cv2.rectangle(image, (x+8*l, y+3*l), (x+9*l, y+4*l), (142,191,251), -1) 

#Brazos color
cv2.rectangle(image, (x+9*l, y+14*l), (x+10*l, y+15*l), (142, 191, 251), -1)
cv2.rectangle(image, (x+8*l, y+15*l), (x+9*l, y+16*l), (142, 191, 251), -1)
cv2.rectangle(image, (x+7*l, y+16*l), (x+8*l, y+17*l), (142, 191, 251), -1)

#torso color
cv2.rectangle(image, (x+13*l, y+15*l), (x+14*l, y+16*l), (189, 112, 2), -1)
cv2.rectangle(image, (x+12*l, y+21*l), (x+14*l, y+22*l), (189, 112, 2), -1)


cv2.rectangle(image, (x+11*l, y+22*l), (x+12*l, y+23*l), (189, 112, 2), -1)
cv2.rectangle(image, (x+10*l, y+23*l), (x+12*l, y+24*l), (189, 112, 2), -1)
cv2.rectangle(image, (x+14*l, y+20*l), (x+16*l, y+15*l), (142, 191, 251), -1)
cv2.rectangle(image, (x+13*l, y+19*l), (x+14*l, y+21*l), (189, 112, 2), -1)
cv2.rectangle(image, (x+12*l, y+20*l), (x+14*l, y+21*l), (189, 112, 2), -1)

#pies color
cv2.rectangle(image, (x+10*l, y+24*l), (x+12*l, y+25*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+10*l, y+25*l), (x+13*l, y+26*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+10*l, y+26*l), (x+12*l, y+27*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+13*l, y+26*l), (x+16*l, y+27*l), (0, 0, 255), -1)

cv2.rectangle(image, (x+17*l, y+26*l), (x+20*l, y+27*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+16*l, y+25*l), (x+19*l, y+26*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+14*l, y+24*l), (x+16*l, y+25*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+13*l, y+23*l), (x+15*l, y+24*l), (0, 0, 255), -1)
cv2.rectangle(image, (x+12*l, y+20*l), (x+13*l, y+22*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+13*l, y+20*l), (x+14*l, y+21*l), (0, 0, 0), -1)
cv2.rectangle(image, (x+14*l, y+19*l), (x+15*l, y+20*l), (0, 0, 0), -1)
    
cv2.imshow('Pixel Art', image)
cv2.waitKey(0)  
cv2.destroyAllWindows()