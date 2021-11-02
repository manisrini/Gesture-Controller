import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, numHands=2, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.numHands = numHands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands(
            self.mode, self.numHands, self.detection_conf, self.tracking_conf
        )
        # for detecting points
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # convert to rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mp_hand.HAND_CONNECTIONS)
        return img

    def findHandPos(self, img, position=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[position]
            # draws points on hand

            for id, lm in enumerate(myHand.landmark):
                height, width, channels = img.shape

                cx = int(lm.x * width)
                cy = int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
