
import cv2
import numpy as np
from comtypes import CLSCTX, CLSCTX_ALL

# import mediapipe as mp

import HandTrackingModule as htm
import math
import time

# imported modules of Andre Miras
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

wCam, hCam = 950, 720 # webcam output window width, height

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionConf= 0.7) # ***
# more detectionConfidance will assure object is actually hand

# code by Andre Miras (pycaw) mit license / all the imported files are above
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# print(volume.GetVolumeRange()) ( -65.25, 0.0 )

volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]


vol = 0 # initially 0
volBar = 400 # initially 400 level volume
volPer = 0 # initially percentage pf volume is 0

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) # ***

    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx , cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length) ( 300, 0 ) range

        # hande range = 50 - 300
        # volume range = -65 - 0

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length) , vol)
        volume.SetMasterVolumeLevel(vol, None)

        if(length < 50):
            cv2.circle(img, (cx, cy), 15, (20, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cv2.putText(img, f'FPS: {int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)} %', (40, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)




    cv2.imshow("Img", img)
    cv2.waitKey(1)