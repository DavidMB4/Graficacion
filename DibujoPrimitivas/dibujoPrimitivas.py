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