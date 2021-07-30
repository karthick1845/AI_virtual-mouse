import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import autopy as ap     #Access an mouse


######################
wCam,hCam=640,480
frameR=100  #Rate reduction
smoothening=7
pTime=0
detector=htm.handDetector(maxHand=1)
wScr,hScr=ap.screen.size()
#print(wScr,hScr)  #1280.0 720.0

######################

pTime=0

plocX,plocY=0,0
clocX,clocY=0,0

cap=cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    #1.Find the hand Landmarks
    success,img=cap.read()
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img)

    # 2. Get the tip of the index and middle finger
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        #print(x1,y1,x2,y2) #Till good

        #3.Check which are up

        fingers=detector.fingersUP()
        #print(fingers)
        cv.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                     (255, 0, 255), 2)


        #4.Only index finger : Moving mode
        if fingers[1]==1 and fingers[2]==0:

            #5.Convert coordinates
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))

            #6.Smoothen a value
            clocX=plocX+(x3-plocX)/smoothening
            clocY=plocY+(y3-plocY)/smoothening

            #7.Simply move a mouse
            ap.mouse.move(wScr-clocX,clocY)
            cv.circle(img,(x1,y1),15,(255,0,255),cv.FILLED)
            plocX,plocY=clocX,clocY

            #8.Both Index and middle fingers are up:Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            #9.Find distance between fingers
            length,img,lineInfo=detector.findDistance(8,12,img)
            print(length)
            # 10.Click mouse if distance short
            if length<40:
                cv.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv.FILLED)

                ap.mouse.click()
            #11.Frame rate

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(img,str(int(fps)),(20,50),cv.FONT_HERSHEY_PLAIN,3,
               (255,0,0),3)
    cv.imshow("IMage",img)
    cv.waitKey(1)
