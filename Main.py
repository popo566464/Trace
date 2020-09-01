import Scanner
import HandTrack 
import ImgWrap 
import cv2
import sys


def main():
    
    dp = 0   #use yolo=1 or opencv=0
 
    
    
    yo = True if dp == 1 else False  
    scannerinit = Scanner.DocSanner()
    scannerinit.scanner_update()
    
    if "oops" in scannerinit.status:
        sys.exit()
    
    handinit = HandTrack.HandControl(weights = "weight\cross-hands.weights", config = "weight\cross-hands.cfg", yolo = yo)
    wrapinit = ImgWrap.WraprespectDoc(f_imgpath ="Scan/0.jpg", b_imgpath = "Scan/1.jpg")
          
    while True:
        
        handcon_frame = handinit.hand_update()
        dirc = handinit.direction
        imgwrap_frame = wrapinit.wrap_update(dirc)
        
        cv2.namedWindow('imgwrap',2)
        cv2.resizeWindow('imgwrap', 480,640)
        cv2.imshow("handcon", handcon_frame)
        cv2.imshow("imgwrap", imgwrap_frame)
        
        if cv2.waitKey(40) == 27:
            break
    handinit.cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
