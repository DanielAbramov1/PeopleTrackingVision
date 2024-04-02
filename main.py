import gloabal_handler as gh
import cv2

# if using a video that is not inside the directory, add the path to the video inside VedioCapture function
cap = cv2.VideoCapture(gh.VIDEO_PATH) # TODO download better quality video

# create an object from stable camera 
object_creator = cv2.createBackgroundSubtractorMOG2() #TODO do its for unstable camera
# object_creator = cv2.createBackgroundSubtractorKNN()


while True:
    ret, frame = cap.read()

    # mask all static object with black
    mask = object_creator.apply(frame)

    # extract contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # clculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > gh.CONTOUR_PIXEL_FILTER:
            cv2.drawContours(frame, [cnt], -1, gh.CONTOUR_COLOR , gh.CONTOUR_THINKNESS)

    cv2.imshow("Mask", mask)    # showing black and white image after masking
    cv2.imshow("Frame",frame)   # showing the video with addint object detection methods

    # Press 'q' to exit the video
    if cv2.waitKey(gh.FPS) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()