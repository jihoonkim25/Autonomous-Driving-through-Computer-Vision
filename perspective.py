import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


def main():
    # cap = cv2.VideoCapture("viral.mp4")
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     r,c = frame.shape[:2]
    #     pts1 = np.float32([[310,650],[1055,650],[570,450],[670,450]])
    #     pts2 = np.float32([[0, 0],[300, 0],[0,300],[300,300]])
    #
    #     M = cv2.getPerspectiveTransform(pts1, pts2)
    #     dst = cv2.warpPerspective(frame, M, (300,300))
    #
    #     plt.subplot(121),plt.imshow(frame),plt.title('Input')
    #     plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #     plt.show()
    #     cv2.waitKey(0)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img_path = os.path.join(os.path.abspath('TEST_SUBJECTS'), 'dash.jpg')
    video = os.path.abspath('lane.mp4')
    photo_pipeline(img_path)
    # video_pipeline(video)
    # new_dst = inv_perspective_warp(filtered_img)
    # contours = find_contours(filtered_img)
    # dipshit = cv2.addWeighted(img, 1, new_dst, 0.5, 0)
    # cv2.polylines(img, contours, True, (0,255,0), 3)
    # cv2.imshow('filtered', img)

    # plt.subplot(121),plt.imshow(rgb_img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dipshit),plt.title('Output')
    # plt.show()

def pedestrian_crossing(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # img = cv2.imread(img_path)
    # dimensions = (720, 1280)
    # resized = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
    (rects, weights) = hog.detectMultiScale(image, winStride=(4,4), padding=(8,8), scale=1.05)
    rect_pts = []
    for i in range(len(weights)):
        if weights[i] >= 0.1:
            rect_pts.append(rects[i])
    # cv2.imshow('bef_ped', image)
    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rect_pts])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    # print(len(pick))
    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,225,0), 2)
    # cv2.imshow('ped', img)
    # cv2.waitKey(0)
    return(image)

def photo_pipeline(img):
    start_time = time.time()
    # img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = perspective_warp(rgb_img)
    filtered_img = filter_image(dst)
    # cv2.imshow('filtered', filtered_img)
    # cv2.waitKey(0)
    out_img, curves, lanes, ploty = sliding_window(filtered_img)
    curverad = get_curve(img, curves[0],curves[1])
    img_ = draw_lanes(img, curves[0], curves[1])
    # final_img = pedestrian_crossing(img_)
    total_time = time.time() - start_time
    return img_, total_time

def video_pipeline(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dst = perspective_warp(rgb_img)
            filtered_img = filter_image(dst)
            # cv2.imshow('filtered', filtered_img)
            # cv2.waitKey(0)
            out_img, curves, lanes, ploty = sliding_window(filtered_img)
            curverad=get_curve(frame, curves[0],curves[1])
            img_ = draw_lanes(frame, curves[0], curves[1])
        except:
            print("ERROR")
        if ret == True:
            cv2.imshow('output', img_)
            if cv2.waitKey(25) %0xFF == ord('q'):
                break


def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0.4,0.8),(0.6,0.8),(0.2,.975),(.7,.975)]),
                     # src=np.float32([(0.3,0.77),(0.67,0.77),(0.3,1),(.67,.95)]),
                     # src=np.float32([(0.3,0.85),(0.7,0.85),(0.25,.975),(.8,.975)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src*img_size
    dst = dst*np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    dst = cv2.warpPerspective(img, M, dst_size)
    return dst

def inv_perspective_warp(img,
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.4,0.8),(0.6,0.8),(0.2,.975),(.7,.975)])
                     # dst=np.float32([(0.3,0.77),(0.67,0.77),(0.3,1),(.67,1)])
                     # dst=np.float32([(0.4,0.6),(0.55,0.6),(0.25,0.975),(.8,.975)])
                     ):

    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    dst = cv2.warpPerspective(img, M, dst_size)
    return dst

def filter_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    final = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
    # cv2.imshow('thresh',thresh)
    return final

def find_white_pixels(image):
    coords = []
    white = (255, 255, 255)
    height, width = image.shape[:2]
    for x in range(width):
        for y in range(height):
            r, g ,b = image[x, y]
            if (r, g, b) == white:
                coords.append((x, y))
    return coords

def find_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print('contours', contours)
    # print('hierarchies', hierarchy)
    # dick = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    # cv2.imshow('contour', dick)
    # cv2.waitKey(0)
    return contours

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist


def sliding_window(img, nwindows=9, margin=200, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = img # np.dstack((img, img, img))*255
    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = 100
    rightx_current = 1148

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(100,255,255), 3)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(100,255,255), 3)
            # cv2.imshow('out_img', out_img)
            # cv2.waitKey(0)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        # if len(good_left_inds) > minpix:
        #     leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        # if len(good_right_inds) > minpix:
        #     rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
        elif len(good_left_inds) > minpix:
            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
        elif len(good_right_inds) > minpix:
            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # print('left_lane_inds',left_lane_inds)
    # print('right_lane_inds',right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)



    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    # print("ploty", len(ploty))
    # print("leftx", len(leftx))
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/720 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (0,200,0))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective



if __name__ == '__main__':
    main()
