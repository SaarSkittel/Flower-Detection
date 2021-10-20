import cv2
import numpy as np
import matplotlib as mt
import math
import os
import glob
from FlowerDetection import FlowerDetection

image = cv2.imread("shutterstock_270569495.jpg", 1)

cols = math.floor(image.shape[1] * 0.5)
rows = math.floor(image.shape[0] * 0.5)

image = cv2.resize(image, (cols, rows))

filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#image = cv2.GaussianBlur(image, (3, 3), 0.5)
#image = cv2.filter2D(image, 0, filter)



img_dir = "Templates"
threshold_templates = 0.5
Rectangles = FlowerDetection.matching_flower(img_dir, threshold_templates, image)
#Rectangles = FlowerDetection.delete_duplicates(Rectangles, Rectangles)


img_dir = "false positive"
threshold_fp = 0.9
Rectanglesfp = FlowerDetection.matching_flower(img_dir, threshold_fp, image)
Rectangles = FlowerDetection.delete_duplicates(Rectangles, Rectanglesfp)
image = FlowerDetection.draw_rectangles(image, Rectangles)
cv2.imshow("img", image)
print(len(Rectangles))
#print(Rectangles)
cv2.imwrite("shutterstock_2705694952.jpg", image)
cv2.waitKey(0)