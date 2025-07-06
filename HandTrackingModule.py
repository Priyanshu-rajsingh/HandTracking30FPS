import cv2
import mediapipe as mp
import time  # to check the frame rate

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = 0.5

        self.mpHands = mp.solutions.hands  # Corrected module name
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands = self.maxHands,
            min_detection_confidence = self.detectionConf,
            min_tracking_confidence = self.trackConf
        )  # Corrected instantiation
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def findPosition(self, image, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # Fixed missing definition

            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 15, (0, 255, 0), cv2.FILLED)  # Fixed color format
        return lmList

def main():
    pTime = 0
    cTime = 0

    VideoObject = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, image = VideoObject.read()
        image = detector.findHands(image)
        lmList = detector.findPosition(image)

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
