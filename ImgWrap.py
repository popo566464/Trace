import cv2
import numpy as np

class WraprespectDoc():
    def __init__(self, f_imgpath, b_imgpath):
        self.img = cv2.imread(f_imgpath)
        self.imgback = cv2.imread(b_imgpath)      
        self.img = cv2.resize(self.img, (480,640))
        self.imgback = cv2.resize(self.imgback, (480,640))
        self.img = cv2.copyMakeBorder(self.img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
        self.imgback = cv2.copyMakeBorder(self.imgback, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
        self.h, self.w = self.img.shape[0:2]
        self.anglex = 0
        self.angley = 0 
        self.anglez = 0 
        self.fov = 42
        self.r = 0
        n = [i for i in range(0,90)]
        n2 = [i for i in range(270,361)]
        n3 = [i for i in range(-90, 1)]
        n4 = [i for i in range(-360, -269)]
        
        m1 = [i for i in range(90,270)]
        m2 = [i for i in range(-269, -89)]
        
        self.force = []
        self.force.extend(n)
        self.force.extend(n2)
        self.force.extend(n3)
        self.force.extend(n4)
        
        self.back = []
        self.back.extend(m1)
        self.back.extend(m2)
        
        self.rotx = 1
        self.roty = 1
        
    def rad(self,x):
        return x * np.pi / 180    

 
    def get_warpR(self):

        global anglex,angley,anglez,fov,h,w,r
        z = np.sqrt(self.h ** 2 + self.w ** 2) / 2 / np.tan(self.rad(self.fov / 2))
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(self.rad(self.anglex)), -np.sin(self.rad(self.anglex)), 0],
                       [0, -np.sin(self.rad(self.anglex)), np.cos(self.rad(self.anglex)), 0 ],
                       [0, 0, 0, 1]], np.float32)
     
        ry = np.array([[np.cos(self.rad(self.angley)), 0, np.sin(self.rad(self.angley)), 0],
                       [0, 1, 0, 0],
                       [-np.sin(self.rad(self.angley)), 0, np.cos(self.rad(self.angley)), 0],
                       [0, 0, 0, 1]], np.float32)
     
        rz = np.array([[np.cos(self.rad(self.anglez)), np.sin(self.rad(self.anglez)), 0, 0],
                       [-np.sin(self.rad(self.anglez)), np.cos(self.rad(self.anglez)), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], np.float32)
     
        self.r = rx.dot(ry).dot(rz)
     
        pcenter = np.array([self.w / 2, self.h / 2, 0, 0], np.float32)
     
        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([self.h, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, self.w, 0, 0], np.float32) - pcenter
        p4 = np.array([self.h, self.w, 0, 0], np.float32) - pcenter
     
        dst1 = self.r.dot(p1)
        dst2 = self.r.dot(p2)
        dst3 = self.r.dot(p3)
        dst4 = self.r.dot(p4)
     
        list_dst = [dst1, dst2, dst3, dst4]
     
        org = np.array([[0, 0],
                        [self.h, 0],
                        [0, self.w],
                        [self.h, self.w]], np.float32)
     
        dst = np.zeros((4, 2), np.float32)
     
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
     
        warpR = cv2.getPerspectiveTransform(org, dst)
        return warpR

    def control(self, hand):  
        if self.anglex >360 or self.anglex < -360 :
                self.anglex = 0       
        if self.angley >360 or self.angley < -360 :
            self.angley = 0    
        c = cv2.waitKey(30)
        if c == ord('w') or "North" in hand:
            self.anglex -= 10
        if c == ord('s') or "South" in hand:
            self.anglex += 10
        if c == ord('a') or "West" in hand:
            self.angley -= 10
        if c == ord('d') or "East" in hand:
            self.angley += 10
            
        if self.angley in self.force:
            self.roty = 1
        if self.anglex in self.force:
            self.rotx = 1    
        if self.angley in self.back:
            self.roty = -1
        if self.anglex in self.back:
            self.rotx = -1   
            
    def wrap_update(self,dir):
        warpR = self.get_warpR()
        if self.rotx* self.roty > 0:    
            result = cv2.warpPerspective(self.img, warpR, (self.w,self.h))
        if self.rotx* self.roty < 0:    
            result = cv2.warpPerspective(self.imgback, warpR, (self.w,self.h))
        self.control(dir)
        return result
            
''' 
init = WraprespectDoc(f_imgpath ="Scan/0.jpg", b_imgpath = "Scan/1.jpg")
while True:
    
    result = init.wrap_update()  
    cv2.namedWindow('result',2)
    cv2.imshow("result", result)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()

''' 
