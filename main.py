import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options,
                                       num_hands = 2,
                                       min_hand_detection_confidence = 0.8,
                                       min_hand_presence_confidence = 0.8,
                                       min_tracking_confidence = 0.8)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb_frame)
    results = detector.detect(mp_image)

    r = g = b = 0

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Different HandLandmarks
            wrist = hand_landmarks[0]
            thumb_cmc = hand_landmarks[1]
            thumb_mcp = hand_landmarks[2]
            thumb_ip = hand_landmarks[3]
            thumb_tip = hand_landmarks[4]
            index_mcp = hand_landmarks[5]
            index_pip = hand_landmarks[6]
            index_dip = hand_landmarks[7]
            index_tip = hand_landmarks[8]
            middle_mcp = hand_landmarks[9]
            middle_pip = hand_landmarks[10]
            middle_dip = hand_landmarks[11]
            middle_tip = hand_landmarks[12]
            ring_mcp = hand_landmarks[13]
            ring_pip = hand_landmarks[14]
            ring_dip = hand_landmarks[15]            
            ring_tip = hand_landmarks[16]
            pinky_mcp = hand_landmarks[17]
            pinky_pip = hand_landmarks[18]
            pinky_dip = hand_landmarks[19]
            pinky_tip = hand_landmarks[20]
            # cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None)

            #Changing the Colour based on the hand
            right = (hand_landmarks[5].x>hand_landmarks[17].x)

            # Line Colour 51, 255, 153
            if right:   # Light Blue for Right Hand
                r = 121
                g = 164
                b = 225
            else:       # Light Orange for Left Hand
                r = 255
                g = 153
                b = 51

            # Drawing Dots on the Tips for the Fingers
            # Wrist
            wrist_x,wrist_y = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame,(wrist_x,wrist_y),5,(b,g,r), -1)

            # Thumb
            tcmc_x,tcmc_y = int(thumb_cmc.x * w), int(thumb_cmc.y * h)
            cv2.circle(frame,(tcmc_x,tcmc_y),5,(b,g,r), -1)
            tmcp_x,tmcp_y = int(thumb_mcp.x * w), int(thumb_mcp.y * h)
            cv2.circle(frame,(tmcp_x,tmcp_y),5,(b,g,r), -1)
            tip_x,tip_y = int(thumb_ip.x * w), int(thumb_ip.y * h)
            cv2.circle(frame,(tip_x,tip_y),5,(b,g,r), -1)
            ttip_x,ttip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            cv2.circle(frame,(ttip_x,ttip_y),5,(b,g,r), -1)

            # Index Finger
            imcp_x,imcp_y = int(index_mcp.x * w), int(index_mcp.y * h)
            cv2.circle(frame,(imcp_x,imcp_y),5,(b,g,r), -1)
            ipip_x,ipip_y = int(index_pip.x * w), int(index_pip.y * h)
            cv2.circle(frame,(ipip_x,ipip_y),5,(b,g,r), -1)
            idip_x,idip_y = int(index_dip.x * w), int(index_dip.y * h)
            cv2.circle(frame,(idip_x,idip_y),5,(b,g,r), -1)
            itip_x,itip_y = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame,(itip_x,itip_y),5,(b,g,r), -1)

            # Middle Finger
            mmcp_x,mmcp_y = int(middle_mcp.x * w), int(middle_mcp.y * h)
            cv2.circle(frame,(mmcp_x,mmcp_y),5,(b,g,r), -1)
            mpip_x,mpip_y = int(middle_pip.x * w), int(middle_pip.y * h)
            cv2.circle(frame,(mpip_x,mpip_y),5,(b,g,r), -1)
            mdip_x,mdip_y = int(middle_dip.x * w), int(middle_dip.y * h)
            cv2.circle(frame,(mdip_x,mdip_y),5,(b,g,r), -1)
            mtip_x,mtip_y = int(middle_tip.x * w), int(middle_tip.y * h)
            cv2.circle(frame,(mtip_x,mtip_y),5,(b,g,r), -1)

            # Ring Finger
            rcmp_x,rcmp_y = int(ring_mcp.x * w), int(ring_mcp.y * h)
            cv2.circle(frame,(rcmp_x,rcmp_y),5,(b,g,r), -1)
            rpip_x,rpip_y = int(ring_pip.x * w), int(ring_pip.y * h)
            cv2.circle(frame,(rpip_x,rpip_y),5,(b,g,r), -1)
            rdip_x,rdip_y = int(ring_dip.x * w), int(ring_dip.y * h)
            cv2.circle(frame,(rdip_x,rdip_y),5,(b,g,r), -1)
            rtip_x,rtip_y = int(ring_tip.x * w), int(ring_tip.y * h)
            cv2.circle(frame,(rtip_x,rtip_y),5,(b,g,r), -1)

            # Pinky Finger
            pmcp_x,pmcp_y = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
            cv2.circle(frame,(pmcp_x,pmcp_y),5,(b,g,r), -1)
            ppip_x,ppip_y = int(pinky_pip.x * w), int(pinky_pip.y * h)
            cv2.circle(frame,(ppip_x,ppip_y),5,(b,g,r), -1)
            pdip_x,pdip_y = int(pinky_dip.x * w), int(pinky_dip.y * h)
            cv2.circle(frame,(pdip_x,pdip_y),5,(b,g,r), -1)
            ptip_x,ptip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
            cv2.circle(frame,(ptip_x,ptip_y),5,(b,g,r), -1)

            # Drawing Connected Lines Between the Key Points
            # From Wrist
            cv2.line(frame,(wrist_x, wrist_y), (tcmc_x,tcmc_y), (102,255,102),2)
            cv2.line(frame,(wrist_x, wrist_y), (imcp_x,imcp_y), (102,255,102),2)
            cv2.line(frame,(wrist_x, wrist_y), (pmcp_x,pmcp_y), (102,255,102),2)
            # Connected in Palm
            cv2.line(frame,(imcp_x, imcp_y), (mmcp_x,mmcp_y), (102,255,102),2)
            cv2.line(frame,(mmcp_x,mmcp_y), (rcmp_x,rcmp_y), (102,255,102),2)
            cv2.line(frame,(rcmp_x,rcmp_y), (pmcp_x,pmcp_y), (102,255,102),2)
            # Thumb Lines
            cv2.line(frame,(tcmc_x,tcmc_y), (tmcp_x,tmcp_y), (102,255,102),2)
            cv2.line(frame,(tmcp_x,tmcp_y), (tip_x,tip_y), (102,255,102),2)
            cv2.line(frame,(tip_x,tip_y), (ttip_x,ttip_y), (102,255,102),2)
            # Index Lines
            cv2.line(frame,(imcp_x, imcp_y), (ipip_x,ipip_y), (102,255,102),2)
            cv2.line(frame,(ipip_x,ipip_y), (idip_x,idip_y), (102,255,102),2)
            cv2.line(frame,(idip_x,idip_y), (itip_x,itip_y), (102,255,102),2)
            # Middle Lines
            cv2.line(frame,(mmcp_x,mmcp_y), (mpip_x,mpip_y), (102,255,102),2)
            cv2.line(frame,(mpip_x,mpip_y), (mdip_x,mdip_y), (102,255,102),2)
            cv2.line(frame,(mdip_x,mdip_y), (mtip_x,mtip_y), (102,255,102),2)
            # Ring Lines
            cv2.line(frame,(rcmp_x,rcmp_y), (rpip_x,rpip_y), (102,255,102),2)
            cv2.line(frame,(rpip_x,rpip_y), (rdip_x,rdip_y), (102,255,102),2)
            cv2.line(frame,(rdip_x,rdip_y), (rtip_x,rtip_y), (102,255,102),2)
            # Pinky Lines
            cv2.line(frame,(pmcp_x,pmcp_y), (ppip_x,ppip_y), (102,255,102),2)
            cv2.line(frame,(ppip_x,ppip_y), (pdip_x,pdip_y), (102,255,102),2)
            cv2.line(frame,(pdip_x,pdip_y), (ptip_x,ptip_y), (102,255,102),2)

    cv2.imshow("Finger tips", frame)      

    #quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()