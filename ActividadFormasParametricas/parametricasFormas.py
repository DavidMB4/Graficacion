import numpy as np
import cv2

width, height = 1000, 1000  
img = np.ones((height, width, 3), dtype=np.uint8)*255

a, b = 150, 100  
k = 1.21
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
        
        cv2.circle(img, (x, y), 2, (39, 247, 0), 2) 
        cv2.circle(img, (x+10, y+10), 2, (137, 247, 0), 2)  

    cv2.imshow("Parametric Animation", img)
    
    theta += theta_increment

    if cv2.waitKey(30) & 0xFF == 27: 
        break

cv2.destroyAllWindows()