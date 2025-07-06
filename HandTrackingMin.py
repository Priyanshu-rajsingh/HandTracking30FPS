# max_num_hands=1 change it at line 12

import cv2
import mediapipe as mp

import time # to check the frame rate

VideoObject = cv2.VideoCapture(0) # 0 - built in camera / 1, 2 .. for external camera

mpHands = mp.solutions.hands
# hands is object of Hands class, used to detect hand
hands = mpHands.Hands(static_image_mode='False', max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

pTime = 0 # previous time
cTime = 0 # current time

while True:
    # Read a frame from the video source (returns success status and the image/frame)
    success, image = VideoObject.read()

    # Convert the image from BGR (OpenCV default) to RGB color format (commonly used in other libraries like PIL, matplotlib, etc.)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks) // coordinates of hand

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                # print(id, lm.x, lm.y)
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    # cv2.circle(img, center, radius, color, thickness)
                    cv2.circle(image, (int(cx), int(cy)), 10, (255,25,166), cv2.FILLED)
                    # the above line draws a circle indicating Hand Land Marks


            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            # use image because its the original input image
            # mpHands.HAND_CONNECTIONS connects the dots for 14 points in hand

    # swap te cTime and pTime utilizing
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # cv2.putText(img, text, org, font, fontScale, color, thickness)
    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=2 )
    cv2.imshow("Image", image)

    # Display the captured frame in a window titled "image"
    cv2.imshow("image", image)

    # Wait 1 ms to allow imshow to update the window; can also detect key presses
    cv2.waitKey(1)



"""
MediaPipe Hand Landmark Points (21 total):

 0  - Wrist

Thumb:
 1  - Thumb_CMC (Carpometacarpal joint)
 2  - Thumb_MCP (Metacarpophalangeal joint)
 3  - Thumb_IP  (Interphalangeal joint)
 4  - Thumb_Tip

Index Finger:
 5  - Index_Finger_MCP
 6  - Index_Finger_PIP
 7  - Index_Finger_DIP
 8  - Index_Finger_Tip

Middle Finger:
 9  - Middle_Finger_MCP
10  - Middle_Finger_PIP
11  - Middle_Finger_DIP
12  - Middle_Finger_Tip

Ring Finger:
13  - Ring_Finger_MCP
14  - Ring_Finger_PIP
15  - Ring_Finger_DIP
16  - Ring_Finger_Tip

Pinky Finger:
17  - Pinky_MCP
18  - Pinky_PIP
19  - Pinky_DIP
20  - Pinky_Tip
"""
