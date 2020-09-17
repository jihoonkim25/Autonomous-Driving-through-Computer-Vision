import cv2
import numpy as np
import os
import math
import imutils
import shutil
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow import image
import time
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import re
import skimage.transform



IMG_WIDTH = 60
IMG_HEIGHT = 60
CONFIDENCE = 0.98
path_to_model = os.path.abspath('udacity_model_v2')
model = keras.models.load_model(path_to_model)

# path_to_img = os.path.join(os.path.abspath('TEST_SUBJECTS'), 'right.jpg')
# original_image = cv2.imread(path_to_img)


def main():
    # contour_crops, coord_dict = contour_crop(original_image)
    # dir = 'Split_Pictures'
    # predict_img(original_image, coord_dict, dir)
    cap = cv2.VideoCapture('galore.mp4')

    while True:
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contour_crops, coord_dict = contour_crop(frame)
        image = predict_img(frame, coord_dict, 'Split_Pictures')
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_img(frame, coord_dict, dir):
    start_time = time.time()
    labels = os.listdir(os.path.abspath('without_small_signs'))
    split_dir = os.path.abspath(dir)
    entries = os.listdir(split_dir)
    images = {}
    for crop in entries:
        crop_path = os.path.join(split_dir, crop)
        img = cv2.imread(crop_path)
        w, h, dim = img.shape
        if img is not None:
            dimensions = (IMG_WIDTH, IMG_HEIGHT)
            resized_pre = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
            resized = np.expand_dims((resized_pre), axis=0)
            predictions = model.predict(resized)
            if max(predictions[0]) >= CONFIDENCE:
                predicted_label = labels[np.argmax(predictions[0])]
                # print('IMAGE: ', crop)
                print('PREDICTED_LABEL: ', predicted_label)
                print('CONFIDENCE VALUE: ', predictions[0][np.argmax(predictions[0])])
                cv2.rectangle(frame, ((coord_dict[crop])[0], (coord_dict[crop])[1]), ((coord_dict[crop])[2], (coord_dict[crop])[3]), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, ((coord_dict[crop])[0], (coord_dict[crop])[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, (15, 15, 178), 2)
    # cv2.imshow('img', original_image)
    # cv2.waitKey(0)
    shutil.rmtree(split_dir)
    total_time = time.time() - start_time
    return(frame, total_time)

def contour_crop(img):
    ''' Takes image and applies filters to detect edges of traffic sign '''
    start_time = time.time()
    images = []
    coord_dict = {}
    # Read picture
    # img = cv2.imread(path_to_img)
    # cv2.imshow('original', img)

    # define dimensions of image and center
    # measurement of picture starts from top left corner
    height, width = img.shape[:2]
    #print(str(height)+" "+str(width))
    center_y = int(height/2)
    center_x = int(width/2)

    # define array of distance from center of image - it connected with area of contour
    # more distance from center - more bigger contour (look at the picture
    # test_1.jpg - 3 red squares shows this areas)
    dist_ = [center_x/3, center_x/2, center_x/1.5]

    # defining main interest zone of picture (left, right, top bottom borders)
    # this zone is approximate location of traffic sings
    # green zone at the picture test_1.jpg
    left_x = 0
    right_x = width
    top_y = 0
    bottom_y = height
    # crop zone of traffic signs location to search only in it
    crop_image = img[top_y:bottom_y, left_x:right_x]
    #cv2.imshow('img0',crop_image)
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Make Canny Image - first image for recognition of shape
    canny_img = cv2.Canny(gray, 50, 240)
    # blur_canny_img = cv2.GaussianBlur(canny_img, (3, 3), 0)
    blur_canny_img = cv2.blur(canny_img,(2,2))
    # blur_canny_img = cv2.medianBlur(canny_img,5)
    _,thresh_canny_img = cv2.threshold(blur_canny_img, 127, 255, cv2.THRESH_BINARY)

    # cv2.imshow('Canny', thresh_canny_img)

    # # Make color HSV image - second image for color mask recognition
    # # Converting BGR to HSV
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # cv2.imshow('HSV', hsv_img)
    # # lower mask (0-10)
    # lower_red = np.array([0,50,50],np.uint8)
    # upper_red = np.array([10,255,255],np.uint8)
    # mask_red_lo = cv2.inRange(hsv_img, lower_red, upper_red)
    # # upper mask (170-180)
    # lower_red = np.array([160,50,50], np.uint8)
    # upper_red = np.array([180,255,255], np.uint8)
    # mask_red_hi = cv2.inRange(hsv_img, lower_red, upper_red)
    # # blue color mask
    # lower_blue=np.array([100,50,50],np.uint8)
    # upper_blue=np.array([140,200,200],np.uint8)
    # mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # # yellow color mask
    # lower_yellow=np.array([15,110,110],np.uint8)
    # upper_yellow=np.array([25,255,255],np.uint8)
    # mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    #
    # # join all masks
    # # could be better to join yellow and red mask first  - it can helps to detect
    # # autumn trees and delete some amount of garbage, but this is TODO next
    # mask = mask_red_lo+mask_red_hi+mask_yellow+mask_blue
    #
    # # find the colors within the specified boundaries and apply
    # # the mask
    # hsv_out = cv2.bitwise_and(hsv_img, hsv_img, mask = mask)
    #
    # #blurred image make lines from points and parts and increase quality (1-3,1-3) points
    # blur_hsv_out = cv2.blur(hsv_out,(2,2)) # change from 1-3 to understand how it works
    #
    # # preparing HSV for countours - make gray and thresh
    # gray = cv2.cvtColor(blur_hsv_out, cv2.COLOR_BGR2GRAY)
    # # increasing intensity of finded colors with 0-255 value of threshold
    # # look at the file test_1_hsv_binary to understand what the file thresh is
    # _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # thresh_inv = ~thresh

    # cv2.imshow('original', img)
    # cv2.imshow('canny', thresh_canny_img)
    # # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)

    ''' Takes filtered images as input and crops traffic signs '''
    multiangles_n=0
    # contours of the first image (thresh_canny_img)
    # cv2.RETR_TREE parameter shows all the contours internal and external
    contours1, hierarchy1 = cv2.findContours(thresh_canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Contours total at first image: "+str(len(contours1)))

    #take only first  biggest 15% of all elements
    #skipping small contours from tree branches etc.
    contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)[:int(len(contours1)/6)]

    for cnt in contours1:
        # find perimeters of area - if it small and not convex - skipping
        perimeter = cv2.arcLength(cnt,True)
        if perimeter< 90 or cv2.isContourConvex=='False':#25 - lower - more objects higher-less
            continue

        #calculating rectangle parameters of contour
        (x,y),(w,h),angle = cv2.minAreaRect(cnt)
        # calculating coefficient between width and height to understand if shape is looks like traffic sign or not
        coeff_p = 0
        if w>=h and h != 0:
            coeff_p = w/h
        elif w != 0:
            coeff_p = h/w
        if coeff_p > 2: # if rectangle is very thin then skip this contour
            continue

        # compute the center of the contour
        M = cv2.moments(cnt)
        cX = 0
        cY = 0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # transform cropped image coordinates to real image coordinates
        cX +=left_x
        cY +=top_y

        dist_c_p = math.sqrt(math.pow((center_x-cX),2) + math.pow((center_y-cY),2))
        # skipping small contours close to the left and right sides of picture

        if dist_c_p > dist_[0] and dist_c_p <= dist_[1] and perimeter < 30:
            continue
        if dist_c_p > dist_[1] and dist_c_p <= dist_[2] and perimeter < 50:
            continue
        if dist_c_p > dist_[2] and perimeter < 70:
            continue
        # 0,15 - try to use different coefficient to better results
        approx_c = cv2.approxPolyDP(cnt,0.15*cv2.arcLength(cnt,True),True) #0,15 - lower - more objects higher-less
        if len(approx_c)>=3: # if contour has more then two angles...
            # calculating parameters of rectangle around contour to crop ROI of porential traffic sign
            x,y,w_b_rect,h_b_rect = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(cX-int(w_b_rect/2)-10,cY-int(h_b_rect/2)-10),(cX+int(w_b_rect/2)+10,cY+int(h_b_rect/2)+10),(255,0,0),1)
            # put this ROI to images array for next recognition
            images.append(img[cY-int(h_b_rect/2)-3:cY+int(h_b_rect/2)+3, cX-int(w_b_rect/2)-3:cX+int(w_b_rect/2)+3])

            name = ('%r_contour_crop.jpg' % multiangles_n)
            coord_dict[name] = cX-int(w_b_rect/2)-10, cY-int(h_b_rect/2)-10, cX+int(w_b_rect/2)+10, cY+int(h_b_rect/2)+10
            if not(cY-int(h_b_rect/2) < 0 or cY+int(h_b_rect/2)+0 > height or cX-int(w_b_rect/2) < 0 or cX+int(w_b_rect/2) > width):
                cv2.imwrite(name,img[cY-int(h_b_rect/2)-0:cY+int(h_b_rect/2)+0, cX-int(w_b_rect/2)-0:cX+int(w_b_rect/2)+0])
            #increasing multiangles quantity
            multiangles_n+=1

    # contours in second image (thresh)
    # in this picture we are only use RETR_EXTERNAL contours to avoid processing for example windows in yellow and red houses
    # and holes between plants etc
    # contours2, hierarchy2 = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # #print("Contours total at second image: "+str(len(contours2)))
    #
    # # make first 10% biggest contours +- of elements
    # contours2 = sorted(contours2, key = cv2.contourArea, reverse = True)[:int(len(contours2)/10)]
    #
    # for cnt in contours2:
    #     #calculating perimeter
    #     perimeter = cv2.arcLength(cnt,True)
    #     # if perimeter id too big or too small and is not convex skipping
    #     if perimeter > 2000 or perimeter<30 or cv2.isContourConvex=='False':#25 - lower - more objects higher-less
    #         continue
    #
    #     #calculating rectangle parameters of contour
    #     (x,y),(w,h),angle = cv2.minAreaRect(cnt)
    #     # calculating coefficient between width and height to understand if shape is looks like traffic sign or not
    #     coeff_p = 0
    #     if w>=h and h != 0:
    #         coeff_p = w/h
    #     elif w != 0:
    #         coeff_p = h/w
    #     if coeff_p > 2: # if rectangle is very thin then skip this contour
    #         continue
    #
    #     # compute the center of the contour
    #     M = cv2.moments(cnt)
    #     cX = 0
    #     cY = 0
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #
    #     # transform cropped image coordinates to real image coordinates
    #     cX +=left_x
    #     cY +=top_y
    #
    #     dist_c_p = math.sqrt(math.pow((center_x-cX),2) + math.pow((center_y-cY),2))
    #     # skipping small contours close to the left and right sides of picture
    #     if dist_c_p > dist_[0] and dist_c_p <= dist_[1] and perimeter < 30:
    #         continue
    #     if dist_c_p > dist_[1] and dist_c_p <= dist_[2] and perimeter < 50:
    #         continue
    #     if dist_c_p > dist_[2] and perimeter < 70:
    #         continue
    #
    #     approx_c = cv2.approxPolyDP(cnt,0.03*cv2.arcLength(cnt,True),True) #0,03 - lower - more objects higher-less
    #     if len(approx_c)>=3:
    #         x,y,w_b_rect,h_b_rect = cv2.boundingRect(cnt)
    #         cv2.rectangle(img,(cX-int(w_b_rect/2)-10,cY-int(h_b_rect/2)-10),(cX+int(w_b_rect/2)+10,cY+int(h_b_rect/2)+10),(0,255,0),1)
    #         images.append(img[cY-int(h_b_rect/2)-3:cY+int(h_b_rect/2)+3, cX-int(w_b_rect/2)-3:cX+int(w_b_rect/2)+3])
    #         name = ('%r_contour_crop.jpg' % multiangles_n)
    #         cropped_img = cv2.imwrite(name,img[cY-int(h_b_rect/2)-3:cY+int(h_b_rect/2)+3, cX-int(w_b_rect/2)-3:cX+int(w_b_rect/2)+3])
    #         coord_dict[name] = cX-int(w_b_rect/2)-10, cY-int(h_b_rect/2)-10, cX+int(w_b_rect/2)+10, cY+int(h_b_rect/2)+10
    #         multiangles_n+=1
    # # print(str(multiangles_n) + ' showed multiangles')

    split_dir = "C:\\Users\\jkim2\\desktop\\CS50\\ezra\\traffic\\Split_Pictures"
    os.mkdir(split_dir)
    # abspath = "C:\\Users\\Team 3256_2\\PycharmProjects\\Practice\\traffic"
    abspath = "C:\\Users\\jkim2\\desktop\\CS50\\ezra\\traffic"
    traffic_dir = os.listdir(abspath)
    jpg_dir = []
    for file in traffic_dir:
        if '.jpg' in file:
            jpg_dir.append(file)
    jpg_dir.sort(key=lambda f: int(re.sub('\D', '', f)))
    width_range = list(range(-999,-998))
    height_range = list(range(-999,-998))
    for file in jpg_dir:
        file_path = abspath + "\\" + file
        if coord_dict[file][0] not in width_range and coord_dict[file][2] not in width_range and coord_dict[file][1] not in height_range and coord_dict[file][3] not in height_range:
            # print(file)
            # print('WIDTH RANGE', width_range)
            # print('HEIGHT RANGE', height_range)
            shutil.move(file_path, split_dir)
        else:
            os.remove(file_path)
        width_range = list(range(coord_dict[file][0], coord_dict[file][2]+1))
        height_range = list(range(coord_dict[file][1], coord_dict[file][3]+1))
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    total_time = time.time() - start_time
    return images, coord_dict, total_time



if __name__ == '__main__':
    main()
