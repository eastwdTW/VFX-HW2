import numpy as np
import cv2
import os

if __name__ == "__main__":
    h, w = 720, 540
    for i in range(1, 4):
        input_path = "../data/sample{}".format(i)
        files = os.listdir(input_path)

        for f in files:
            img = cv2.imread(input_path + "/" + f)
            img = cv2.resize(img, (w, h))
            cv2.imwrite(input_path + "/" + f, img)