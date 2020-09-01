import cv2
import numpy as np
from collections import deque

class HandControl():
    def __init__(self, weights =" ", config =" ", yolo = False):
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        self.yolo = yolo
        
        if self.yolo:
            self.net = cv2.dnn.readNet(weights , config)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.output_layers = self.net.getUnconnectedOutLayersNames()       
            self.colors =  np.random.randint(0,255,size=(1, 3))
        else:
            self.frame1 = self.getCameraRead()
        
        self.trackLength = 32
        self.directionPoints = 10
        
        self.pts = deque(maxlen=self.trackLength)
        self.direction = ""
        (self.dX, self.dY) = (0, 0)
        (self.centers_x, self.centers_y) = (0,0)
    
    def getCameraRead(self):
        _,frame = self.cap.read()
        return cv2.flip(frame, 1)
    
    def yolo_hand_detect(self,image):
        #global centers_x, centers_y
        height, width, channels = image.shape
    
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        centers = []
    
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    centers.append([center_x, center_y])
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                self.centers_x, self.centers_y = centers[i]
                color= [int(c) for c in self.colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        return (self.centers_x, self.centers_y)
    
    def opencv_hand_track(self, t0, t1):
        c1 = cv2.cvtColor(t0, cv2.COLOR_BGR2LAB)
        c2 = cv2.cvtColor(t1, cv2.COLOR_BGR2LAB)
        _, grey1, _  = cv2.split(c1)
        _, grey2, _ = cv2.split(c2)
    
        d = cv2.absdiff(grey1, grey2)
        d = cv2.GaussianBlur(d,(5,5),0)
    
        #cv2.imshow("Gray", d)
        #cv2.imwrite("gray-"+str(int(time.time()))+".jpg", d)
    
        ret, mask = cv2.threshold( d, 8, 255, cv2.THRESH_BINARY )
        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=10)
        cv2.imshow("Mask", mask)
        #cv2.imwrite("mask-"+str(int(time.time()))+".jpg", mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        areas = [cv2.contourArea(c) for c in cnts]
    
        if(len(areas)>0):    
            max_index = np.argmax(areas)
            cnt=cnts[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            if(areas[max_index]>5000):
                cv2.rectangle(t1,(x,y),(x+w,int(y+w*1.35)), (0,255,0),2)
                #M = cv2.moments(cnt)
                #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                center = (int( x+(w/2)), y+30)
        return center
      

    def get_key (self, dictory, value):
        return [k for k, v in dictory.items() if v == value]
    
    def grepObject(self,t_img, center):
        #global pts, direction, dX, dY
    
        self.pts.append(center)
    
        for ii in np.arange(1, len(self.pts)):
            i = len(self.pts) - ii
    
            if(len(self.pts)> self.directionPoints):
    
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue            
                elif self.pts[-self.directionPoints] is not None:
                    self.dX = self.pts[-self.directionPoints][0] - self.pts[i][0]
                    self.dY = self.pts[-self.directionPoints][1] - self.pts[i][1]
                    #(dirX, dirY) = ("", "")
                    
                    dic = {"dx" : self.dX, "dy" : self.dY}
                    tempp = (self.dX, self.dY)
                    tmp = tempp[np.argmax(np.abs(tempp))]
                    
                    temp_dire = self.get_key(dic, tmp)
                    
                    if self.dX ==0 or self.dY ==0:
                        self.direction =" "

                    if "dx" in temp_dire :
                        if(np.abs(self.dX)) > 0:
                            if self.dX > 0: 
                                self.direction = "East"
                            elif self.dX <0: 
                                self.direction = "West"
                                
                    elif "dy" in temp_dire :
                        if(np.abs(self.dY)) > 0:
                            if self.dY >0 : 
                                self.direction = "South"
                            elif self.dY <0: 
                                self.direction = "North"
                        
            thickness = int(np.sqrt(self.trackLength/(float(ii + 1))) * 7.2)
            cv2.line(t_img, self.pts[i - 1], self.pts[i], (0, 250, 253), thickness)
    
        self.draw_text(t_img)
    
        return t_img
    
    def draw_text(self, image):
        cv2.putText(image, "West", (10, image.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)
        cv2.putText(image, "East", (image.shape[1]-40, image.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)
        cv2.putText(image, "North", (image.shape[1]//2, 20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)
        cv2.putText(image, "South", (image.shape[1]//2, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)
    
        cv2.putText(image, self.direction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
            1.55, (0, 0, 255), 3)
        cv2.putText(image, "dx: {}, dy: {}".format(self.dX, self.dY),
            (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    def hand_update(self):
        img = self.getCameraRead()
        
        if self.yolo:           
            center_coor = self.yolo_hand_detect(img)
        else:
            center_coor = self.opencv_hand_track(self.frame1, img)
            
        frame = self.grepObject(img, center_coor)
        
        if self.yolo == False:
            if cv2.waitKey(1) == 32:
                self.frame1 = self.getCameraRead()
        return frame

'''
#init = YoloControl(weights = "weight\cross-hands.weights", config = "cross-hands.cfg", yolo=True)
init = HandControl()

while True:    
    img = init.hand_update()    
    cv2.imshow("Image", img)
    if cv2.waitKey(40) == 27:
        break
init.cap.release()
cv2.destroyAllWindows()
'''
