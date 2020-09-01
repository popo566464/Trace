import cv2
import numpy as np

class DocSanner():
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(10,160)
        self.heightImg = 640
        self.widthImg  = 480
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        
        self.status = ""
        self.initializeTrackbars()
        self.count=0


    def stackImages(self, imgArray, scale, lables=[]):
        rows = len(imgArray)   #2
        cols = len(imgArray[0])  #4
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]  #640
        height = imgArray[0][0].shape[0] #480
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: 
                        imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
    
        if len(lables) != 0:
            eachImgWidth= int(ver.shape[1] / cols)
            eachImgHeight = int(ver.shape[0] / rows)
            for d in range(0, rows):
                for c in range (0,cols):
                    cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                    cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
        return ver

    def reorder(self, myPoints):
    
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        center = (myPoints.sum(0)[0] /4, myPoints.sum(0)[1]/4)
            
        up =[i for i in myPoints if i[1]<center[1] ]
        down = [i for i in myPoints if i[1]>center[1] ]
        
        up_left = up[np.argmin(up, axis=0)[0]]
        up_right =up[np.argmax(up, axis=0)[0]]  
        down_left = down[np.argmin(down, axis=0)[0]]
        down_right = down[np.argmax(down, axis=0)[0]]
         
        myPointsNew[0] = up_left
        myPointsNew[1] = up_right
        myPointsNew[2] = down_left
        myPointsNew[3] = down_right
     
        return myPointsNew

    def biggestContour(self, contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area
    
    def drawRectangle(self, img,biggest,thickness):
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    
        return img
    
    def nothing(self, x):
        pass
    
    def initializeTrackbars(self, intialTracbarVals=0):
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 360, 240)
        cv2.createTrackbar("Threshold1", "Trackbars", 100, 255, self.nothing)
        cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, self.nothing)
    
    
    def valTrackbars(self):
        Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
        Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
        src = Threshold1,Threshold2
        return src
    
    def scanner_update(self):       
        while True:       
            success, img = self.cap.read()            
            img = cv2.resize(img, (self.widthImg, self.heightImg))
            imgBlank = np.zeros((self.heightImg,self.widthImg, 3), np.uint8) 
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
            thres=self.valTrackbars() 
            imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])
            imgDial = cv2.dilate(imgThreshold, self.kernel, iterations=2)
            imgThreshold = cv2.erode(imgDial, self.kernel, iterations=1)  
        
            imgContours = img.copy() 
            imgBigContour = img.copy() 
            contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 
        
            biggest, maxArea = self.biggestContour(contours) 
            
            if biggest.size != 0:   
                biggest=self.reorder(biggest)
                cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10) 
                imgBigContour = self.drawRectangle(imgBigContour,biggest,2)
                pts1 = np.float32(biggest) 
                pts2 = np.float32([[0, 0],[self.widthImg, 0], [0, self.heightImg],[self.widthImg, self.heightImg]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (self.widthImg, self.heightImg))
        
                imgWarpColored=imgWarpColored[10:imgWarpColored.shape[0] - 10, 10:imgWarpColored.shape[1] - 10]
                imgWarpColored = cv2.resize(imgWarpColored,(self.widthImg, self.heightImg))
                
                self.status = "success"
            else:
                imgBigContour = imgBlank
                imgWarpColored = imgBlank
                
                self.status = "oops"
                
            imageArray = ([img,imgContours],
                          [imgBigContour, imgWarpColored])
                
 
            lables = [["Original","Contours"],
                      ["Biggest Contour","Warp Prespective"]]            
            
            stackedImage = self.stackImages(imageArray,0.75,lables)
            cv2.imshow("Result",stackedImage)
        
            if cv2.waitKey(1) == 32 :
                cv2.imwrite("Scan/"+str(self.count)+".jpg",imgWarpColored)
                cv2.putText(stackedImage, self.status, (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                            cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 5, cv2.LINE_AA)
                cv2.imshow('Result', stackedImage)
                cv2.waitKey(300)
                break
        self.cap.release()
        cv2.destroyAllWindows()
        
#init = DocSanner()
#init.scanner_update()
        
        
        
        