# import cv2
# import imutils
  
# # Initializing the HOG person
# # detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
  
# cap = cv2.VideoCapture('video_short.mp4')
  
# while cap.isOpened():
#     # Reading the video stream
#     ret, image = cap.read()
#     if ret:
#         image = imutils.resize(image, 
#                                width=min(400, image.shape[1]))
  
#         # Detecting all the regions 
#         # in the Image that has a 
#         # pedestrians inside it
#         (regions, _) = hog.detectMultiScale(image,
#                                             winStride=(4, 4),
#                                             padding=(4, 4),
#                                             scale=1.05)
  
#         # Drawing the regions in the 
#         # Image
#         for (x, y, w, h) in regions:
#             cv2.rectangle(image, (x, y),
#                           (x + w, y + h), 
#                           (0, 0, 255), 2)
  
#         # Showing the output Image
#         cv2.imshow("Image", image)
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break
#     else:
#         break
 
# cap.release()
# cv2.destroyAllWindows()

#-------------------------works---------------------------------

import cv2
import imutils

# frame per second value
FPS  = 30

# Predefined colors for visualization
COLORS = [(255, 0, 0), (130, 48, 219)]  # Blue, Pink

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('video_short.mp4')

person_colors = {}  # Dictionary to map person ID to color index

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))

        # Detecting all the regions in the image with pedestrians inside it
        (regions, _) = hog.detectMultiScale(image,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)

        # Drawing the regions in the image
        for i, (x, y, w, h) in enumerate(regions):
            person_id = f'Person_{i}'  # Unique identifier for each person
            if person_id not in person_colors:
                person_colors[person_id] = i % len(COLORS)  # Assign color index based on the number of predefined colors

            color_index = person_colors[person_id]
            color = COLORS[color_index]
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          color, 2)

        cv2.imshow("Image", image)

        # if 'q' is pressed exit all
        if cv2.waitKey(FPS) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
