import cv2

image = cv2.imread('right2.jpg')

flipped = cv2.flip(image,1)

cv2.imwrite('right2_flip.jpg',flipped)