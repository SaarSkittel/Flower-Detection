import glob
import os

import cv2
import numpy as np
import math
import copy

class FlowerDetection:

    @staticmethod
    def matching(image, template, threshold):
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #template = cv2.GaussianBlur(template, (3, 3), 0.5)
        #template = cv2.filter2D(template, 0, filter)
        temp_w = template.shape[1]
        temp_h = template.shape[0]
        match = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) #find matching result by threshold value
        location = np.where(match >= threshold) # finds locations with the matching value of desired threshold value
        matchLoc = list(zip(*location[::-1])) #list of location
        Rectangles = []
        for loc in matchLoc:# create list of ractangles with the locations an template size
            rect = [int(loc[0]), int(loc[1]), temp_w, temp_h]
            Rectangles.append(rect)
        Rectangles, weight = cv2.groupRectangles(Rectangles, 1, 0.5) #delete over laping rectangles
        return Rectangles

    @staticmethod
    def template_prepare(template, size):
        cols = math.floor(template.shape[1] * size)  # reduced columns size
        rows = math.floor(template.shape[0] * size)  # reduced rows size
        template = cv2.resize(template, (cols, rows))
        return template

    @staticmethod
    def merge_list(list1, list2):
        for a in list2:
            rect = [a[0], a[1], a[2], a[3]]
            list1.append(rect)
        return list1

    @staticmethod
    def delete_duplicates(list1, list2):
        a = False
        if list1 != list2:
            a = True
        for l2 in list2:
            x2 = (l2[0]+l2[2]+l2[0])/2
            y2 = (l2[1]+l2[3]+l2[1])/2
            for l1 in list1:
                x1 = (l1[0]+l1[2]+l1[0])/2
                y1 = (l1[1]+l1[3]+l1[1])/2
                if a & (l1 == l2):#if there are the same size and location and not the same lists
                    l1[0] = 0
                    l1[1] = 0
                    l1[2] = 0
                    l1[3] = 0
                elif (l1 == l2) & (a == False):#if it is the same list it will not delete the same rectangle
                    l1[0] = l1[0]
                elif ((x2 >= l1[0]) & (x2 <= l1[0]+l1[2])) & ((y2 >= l1[1]) & (y2 <= l1[1]+l1[3])):
                    #if the center of l2 is inside l1
                    l1[0] = 0
                    l1[1] = 0
                    l1[2] = 0
                    l1[3] = 0
                elif ((x1 >= l2[0]) & (x1 <= l2[0]+l2[2])) & ((y1 >= l2[1]) & (y1 <= l2[1]+l2[3])):
                    # if the center of l1 is inside l2
                    l1[0] = 0
                    l1[1] = 0
                    l1[2] = 0
                    l1[3] = 0
        list = []
        for l in list1:
            if l[2] != 0:
                list.append(l)
        return list

    @staticmethod
    def matching_flower(path, threshold, image):
        data_path = os.path.join(path, '*g')
        files = glob.glob(data_path)
        Rectangles = []

        for file in files:
            template = cv2.imread(file, 1)
            cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(template, template, 0, 255, cv2.NORM_MINMAX)

            Rectangles1 = FlowerDetection.matching(image, template, threshold)
            Rectangles = FlowerDetection.merge_list(Rectangles, Rectangles1)

            res_temp = FlowerDetection.template_prepare(template, 0.75)#reduces in 75%
            Rectangles2 = FlowerDetection.matching(image, res_temp, threshold)
            Rectangles = FlowerDetection.merge_list(Rectangles, Rectangles2)

            res_temp = FlowerDetection.template_prepare(template, 0.5)#reduces in 50%
            Rectangles3 = FlowerDetection.matching(image, res_temp, threshold)
            Rectangles = FlowerDetection.merge_list(Rectangles, Rectangles3)
        return Rectangles

    @staticmethod
    def draw_rectangles(image, rect_list):
        for (x, y, w, h) in rect_list:
            topLeft = (x, y)
            bottomRight = (x + w, y + h)
            cv2.rectangle(image, topLeft, bottomRight, 255, 2)
        return image