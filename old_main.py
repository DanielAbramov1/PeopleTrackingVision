import gloabal_handler as gh
import cv2
from tracker import *

# if using a video that is not inside the directory, add the path to the video inside VedioCapture function
cap = cv2.VideoCapture(gh.VIDEO_PATH) # TODO download better quality video

# create an object from stable camera 
# object_creator = cv2.createBackgroundSubtractorMOG2()
object_creator = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=10) 
# object_creator = cv2.createBackgroundSubtractorKNN()


# create tracker
tracker = EuclideanDistTracker()

while True:
    ret, frame = cap.read()

    # check the video sizes
    if gh.VIDEO_SIZE_PRINT:
        height, width, _ = frame.shape
        print(height,width)

    # extract region of interest
    roi = frame[gh.VIDEO_HEIGHT_L_LIMIT: gh.VIDEO_HEIGHT_H_LIMIT
                ,gh.VIDEO_WIDTH_L_LIMIT: gh.VIDEO_WIDTH_H_LIMIT] # TODO maybe width needs adjustment

    # mask all static object with black
    mask = object_creator.apply(frame)
    _, mask = cv2.threshold(mask, gh.MASK_LOW_THRSLD, gh.MASK_HIGH_THRSLD, cv2.THRESH_BINARY)

    # extract contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # clculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > gh.CONTOUR_PIXEL_FILTER:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), gh.CONTOUR_COLOR, gh.CONTOUR_THINKNESS)
            # cv2.drawContours(frame, [cnt], -1, gh.CONTOUR_COLOR , gh.CONTOUR_THINKNESS)
            # cv2.drawContours(roi, [cnt], -1, gh.CONTOUR_COLOR , gh.CONTOUR_THINKNESS) #BUG changing contour position in original video
            detections.append([x,y,w,h])

    cv2.imshow("Mask", mask)    # showing black and white image after masking
    cv2.imshow("Frame",frame)   # showing the video with addint object detection methods
    cv2.imshow("ROI", roi)

    # Press 'q' to exit the video
    if cv2.waitKey(gh.FPS) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()