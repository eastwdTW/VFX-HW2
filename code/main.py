import numpy as np
import cv2
import argparse
import os
import time
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser("VFX HW2 Image Stitching")
parser.add_argument("-s", default=0, choices=[0, 1, 2], type=int, help="Select the images to be stitched")
parser.add_argument("-r", default=False, type=bool, help="Show feature points and matches for report")
parser.add_argument("--indir", default="../data", type=str, help="The directory where input images are")
parser.add_argument("--type", default="jpg", type=str, help="The data type of input images")
parser.add_argument("--outdir", default="../data/result", type=str, help="The directory where results to be saved")

if __name__ == "__main__":

    args = vars(parser.parse_args())

    print("==========START==========")
    t0 = time.time()
    seed = [101, 40, 99]
    samples = [200, 125, 250]
    overlap = [0.5, 0.8, 0.5]
    mode = [0, 1, 0]

    # read images
    print("Reading images...")

    input_path = args["indir"] + "/sample{}".format(args["s"])
    files = os.listdir(input_path)
    def sortKey(e):
        return int(e[:-4])
    files.sort(key=sortKey)

    imgs = []
    for f in files:
        img = cv2.imread(input_path + "/" + f)
        img[img == 0] = 1
        imgs.append(img)
    imgs = np.array(imgs)
    n, h, w, c = imgs.shape
    print("Done!")
    t1 = time.time()
    print("time cost: {:.2f} sec".format(t1-t0))
    print("\n-------------------------")

    # feature detection
    print("Detecting features...")

    feature_points = []
    descriptors = []
    for i in tqdm(range(n)):
        feature_point, descriptor = HarrisCorner(imgs[i], samples[args["s"]])
        feature_points.append(feature_point)
        descriptors.append(descriptor)
    
    print("Done!")
    t2 = time.time()
    print("time cost: {:.2f} sec".format(t2-t1))
    print("\n-------------------------")

    # feature matching
    print("Matching features...")

    neighbor_matches = []
    for i in tqdm(range(n-1)):
        neighbor_matches.append(MatchFeature(feature_points[i], descriptors[i], feature_points[i+1], descriptors[i+1], w, overlap[args["s"]]))
    
    print("Done!")
    t3 = time.time()
    print("time cost: {:.2f} sec".format(t3-t2))
    print("\n-------------------------")

    # image matching
    print("Matching images...")

    H = []
    for i in tqdm(range(n-1)):
        H.append(Ransac(neighbor_matches[i], seed[args["s"]]))

    print("Done!")
    t4 = time.time()
    print("time cost: {:.2f} sec".format(t4-t3))
    print("\n-------------------------")

    # blending
    print("Blending images...")
    
    last_H = np.eye(3)
    center = n // 2
    result = imgs[center]
    offset_h = 0
    offset_w = 0
    corners = []
    for i in tqdm(range(2*center)):
        if i < center:
            last_H = np.matmul(H[center-1-i], last_H)
            result, offset_h, offset_w = Blend(result, imgs[center-1-i], np.linalg.inv(last_H), offset_h, offset_w)
            if i == center-1:
                corner = np.array([[0, h-1], [0, 0], [1, 1]])
                corner = np.matmul(np.linalg.inv(last_H), corner)
                corner = corner / corner[-1]
                corner[0] = np.ceil(corner[0])
                corner[1] = np.ceil(corner[1] + offset_w)
                corner = corner.astype(np.int32)
                corners.append((corner[0, 0], corner[1, 0]))
                corners.append((corner[0, 1], corner[1, 1]))
        elif i == center:
            last_H = H[i]
            result, offset_h, offset_w = Blend(result, imgs[i+1], last_H, offset_h, offset_w)
        else:
            last_H = np.matmul(last_H, H[i])
            result, offset_h, offset_w = Blend(result, imgs[i+1], last_H, offset_h, offset_w)
            if i == 2*center-1:
                for j in range(len(corners)):
                    newx, newy = corners[j]
                    corners[j] = (newx + offset_h, newy)
                corner = np.array([[0, h-1], [w-1, w-1], [1, 1]])
                corner = np.matmul(last_H, corner)
                corner = corner / corner[-1]
                corner[0] = np.ceil(corner[0] + offset_h)
                corner[1] = np.ceil(corner[1] + offset_w)
                corner = corner.astype(np.int32)
                corners.append((corner[0, 0], corner[1, 0]))
                corners.append((corner[0, 1], corner[1, 1]))

    print("Done!")
    t5 = time.time()
    print("time cost: {:.2f} sec".format(t5-t4))
    print("\n-------------------------")

    # rectangling
    print("Rectangling...")

    unrec_result = np.copy(result)
    uncropped_result, result = rectangling(result, corners, n, h, w, mode[args["s"]])

    print("Done!")
    t6 = time.time()
    print("time cost: {:.2f} sec".format(t6-t5))
    print("\n-------------------------")

    # output result
    print("Writing results...")

    output_path = args["outdir"] + "/sample{}".format(args["s"])

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    cv2.imwrite(output_path + "/result.png", result.astype(np.uint8))
    
    if args["r"]:
        print("Ploting feature points...")
        for i in tqdm(range(n)):
            test = np.copy(imgs[i])
            for x, y in feature_points[i]:
                test[x-2:x+3, y-2:y+3, :] = [0, 0, 255]
            cv2.imwrite(output_path + "/fp{}.png".format(i), test.astype(np.uint8))
        print("Ploting matches...")
        for i in tqdm(range(n-1)):
            test1 = np.copy(imgs[i])
            test2 = np.copy(imgs[i+1])
            match = neighbor_matches[i][0]
            test1[match[0][0]-2:match[0][0]+3, match[0][1]-2:match[0][1]+3, :] = [0, 0, 255]
            test2[match[1][0]-2:match[1][0]+3, match[1][1]-2:match[1][1]+3, :] = [0, 0, 255]
            match = neighbor_matches[i][1]
            test1[match[0][0]-2:match[0][0]+3, match[0][1]-2:match[0][1]+3, :] = [0, 255, 255]
            test2[match[1][0]-2:match[1][0]+3, match[1][1]-2:match[1][1]+3, :] = [0, 255, 255]
            match = neighbor_matches[i][2]
            test1[match[0][0]-2:match[0][0]+3, match[0][1]-2:match[0][1]+3, :] = [255, 0, 0]
            test2[match[1][0]-2:match[1][0]+3, match[1][1]-2:match[1][1]+3, :] = [255, 0, 0]
            match = neighbor_matches[i][3]
            test1[match[0][0]-2:match[0][0]+3, match[0][1]-2:match[0][1]+3, :] = [255, 0, 255]
            test2[match[1][0]-2:match[1][0]+3, match[1][1]-2:match[1][1]+3, :] = [255, 0, 255]
            cv2.imwrite(output_path + "/match{}_0.png".format(i), test1.astype(np.uint8))
            cv2.imwrite(output_path + "/match{}_1.png".format(i), test2.astype(np.uint8))
        print("Writing result without rectangling...")
        cv2.imwrite(output_path + "/unrec.png", unrec_result.astype(np.uint8))
        if mode[args["s"]] == 0:
            print("Writing result without cropping...")
            cv2.imwrite(output_path + "/uncropped.png", uncropped_result.astype(np.uint8))
    
    print("Done!")
    t7 = time.time()
    print("time cost: {:.2f} sec".format(t7-t6))
    print("\n-------------------------")

    print("All done! total time cost: {:.2f} sec".format(t7-t0))
    print("========Complete!========")