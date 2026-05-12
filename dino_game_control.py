import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading


base = python.BaseOptions(model_asset_path= "hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options= base,
    num_hands= 1,
    min_tracking_confidence= 0.7,
    min_hand_detection_confidence= 0.7,
    min_hand_presence_confidence= 0.7,
)

detector = vision.HandLandmarker.create_from_options(options)

game_ready = False

def open_dino():
    global game_ready
    import pyautogui as p
    import time as t
    p.FAILSAFE = True

    p.press("win")
    t.sleep(2)
    p.write("brave")
    t.sleep(2)
    p.press('enter')
    t.sleep(2)
    p.write("brave://dino")
    p.press('enter')
    t.sleep(3)  
    p.click(500,300)  
    
    
    t.sleep(1)
    game_ready = True

def press_space(): 
    import pyautogui as p
    p.press('space')

cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("window", cv2.WND_PROP_TOPMOST, 1)    
cap = cv2.VideoCapture(0)

import time
last_jump_time = 0
cooldown = 0.3
browser_opened = False

while cap.isOpened():
    r,fps = cap.read()   
    if r == True:
        fps = cv2.resize(fps,(500,500))
        fps = cv2.flip(fps,1)

        rgb_fps = cv2.cvtColor(fps,cv2.COLOR_BGR2RGB)
        mp_fps = mp.Image(image_format=mp.ImageFormat.SRGB,data= rgb_fps)

        detection_result = detector.detect(mp_fps)

        if detection_result.hand_landmarks:
            hand = detection_result.hand_landmarks[0]
            x1 = int(hand[4].x * fps.shape[1])
            y1 = int(hand[4].y * fps.shape[0])

            x2 = int(hand[3].x * fps.shape[1])
            y2 = int(hand[3].y * fps.shape[0])

            x3 = int(hand[8].x * fps.shape[1])
            y3 = int(hand[8].y * fps.shape[0])

            x4 = int(hand[7].x * fps.shape[1])
            y4 = int(hand[7].y * fps.shape[0])

            cv2.line(fps,(x1,y1),(x2,y2),(255,255,255),4)
            cv2.circle(fps,(x3,y3),3,(0,0,255),-1)

            
            if y3 < y4:
                if not browser_opened:
                    browser_opened = True
                    threading.Thread(target=open_dino).start()
                elif game_ready:
                    current_time = time.time()
                    if current_time - last_jump_time > cooldown:
                        threading.Thread(target=press_space).start()
                        last_jump_time = current_time
                        
            
            if game_ready and (time.time() - last_jump_time <= 0.2):
                cv2.putText(fps, "JUMP", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("window",fps)


        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    else:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()