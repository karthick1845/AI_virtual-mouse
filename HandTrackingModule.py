
""""
Download from my another hand tracking project
and some change it.

"""

#Module is use for using an handtracking code for various purpose
#In this project Handtracking code will work in With in class.So it can use for any other related projects

import cv2 as cv
import mediapipe as mp
import time
import math

#1:Accessing my webcam

#Importing hand model

class handDetector():
    def __init__(self,mode=False,maxHand=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHand=maxHand
        self.detectionCon=detectionCon
        self.trackingCon=trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHand,self.detectionCon,
                                        self.trackingCon)  # Hand tracking model
        self.mpDraw = mp.solutions.drawing_utils  # Han# d drawing (Joining a points)model
        self.tipIds=[4,8,12,16,20]

    def findHands(self,img,draw=True):
        #Put Fps
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for i in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,i,self.mpHands.HAND_CONNECTIONS)  #mp.hand_connection use for draw a points

                #We dont need landmarks
        return img

    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id,lm) #Converting to pixel
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                #print(id, cx, cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    # IF you want any hand landmark number put that
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList,bbox

    def fingersUP(self):
        fingers=[]
        #Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4Fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector=handDetector()

    # This object(mediapipe)use RGB images so we need to convert it

    while True:
        session, img = cap.read()
        img = detector.findHands(img)
        lmList,bbox=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4]) #Any number you want finger number
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_SIMPLEX, 2,
                   (0, 255, 255), 1)
        cv.imshow("Image", img)

        cv.waitKey(1)



if __name__== "__main__":
    main()