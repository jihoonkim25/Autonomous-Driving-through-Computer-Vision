from perspective import photo_pipeline
from filter import predict_img, contour_crop
import cv2
import os

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_path = os.path.abspath('lane.mp4')
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contour_crops, coord_dict, contour_time = contour_crop(frame)
    # print('CONTOUR_TIME:', contour_time)
    image, predict_time = predict_img(frame, coord_dict, 'Split_Pictures')
    # print('PREDICT_TIME:', predict_time)
    try:
        image, photo_time = photo_pipeline(image)
        # print('PHOTO_TIME:', photo_time)
    except:
        print('photo_pipeline error')
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
