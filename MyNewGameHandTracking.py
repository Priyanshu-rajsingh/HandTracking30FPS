import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

def main():

    VideoObject = cv2.VideoCapture(0)

    detector = htm.handDetector()

    while True:
        success, image = VideoObject.read()
        image = detector.findHands(image)
        lmList = detector.findPosition(image, draw=False)

        if len(lmList) != 0:
            print(lmList[0])  # Thumb location

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    VideoObject.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()