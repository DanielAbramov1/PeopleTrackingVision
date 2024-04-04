import cv2
import imutils

# frame per second value
FPS = 30

# Predefined colors for visualization
COLORS = [(255, 0, 0), (130, 48, 219)]  # Blue, Pink

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('video.mp4')

# Define start and end time in seconds
start_time = 3 * 60 + 18  # Convert 3 minutes and 18 seconds to seconds
end_time = 4 * 60 + 33  # Convert 4 minutes and 33 seconds to seconds

# Calculate the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Set the start and end frames based on the time range
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Flag to indicate if tracking has started
tracking_started = False

# Initialize frame counter
frame_count = 0

# Dictionary to map person ID to color index
person_colors = {}  

while cap.isOpened():
    ret, image = cap.read()
    if not ret :
        break
    
    image = imutils.resize(image, width=min(400, image.shape[1]))

    # Start tracking when the start frame is reached
    if frame_count >= start_frame:
        tracking_started = True

    # End tracking when the end frame is reached
    if(frame_count >= end_frame):
        tracking_started = False

    if tracking_started:
        # Detecting all the regions in the image with pedestrians inside itq
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

    # If 'q' is pressed exit
    if cv2.waitKey(FPS) & 0xFF == ord('q'):
        break

    frame_count += 1


cap.release()
cv2.destroyAllWindows()
