import cv2
import numpy as np
#Read image
image = cv2.imread("images/example_image.jpg")
#Create a dummy image that stores different contrast and brightness
# new_image = np.zeros(image.shape, image.dtype)
#Brightness and contrast parameters
# contrast = 3.0
# bright = 2
#Change the contrast and brightness
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         for c in range(image.shape[2]):
#             new_image[y,x,c] = np.clip(contrast*image[y,x,c] + bright, 0, 255)

#Define font
font= cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(image, "I am a Cat", (230, 50), font, 0.8, (0, 255, 0),2, cv2.LINE_AA)

cv2.imshow('before', image)
# cv2.imshow('after', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()