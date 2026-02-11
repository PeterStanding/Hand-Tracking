import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

base_options = python.BaseOptions(model_asset_path = 'hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options,
                                       num_hands = 1,
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

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]            
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]
            # cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
            # Thumb
            x,y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            cv2.circle(frame,(x,y),5,(255,0,0), -1)
            # Index Finger
            x,y = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame,(x,y),5,(0,255,0), -1)
            # Middle Finger
            x,y = int(middle_tip.x * w), int(middle_tip.y * h)
            cv2.circle(frame,(x,y),5,(0,0,255), -1)
            # Ring Finger
            x,y = int(ring_tip.x * w), int(ring_tip.y * h)
            cv2.circle(frame,(x,y),5,(255,0,255), -1)
            # Pinky Finger
            x,y = int(pinky_tip.x * w), int(pinky_tip.y * h)
            cv2.circle(frame,(x,y),5,(255,255,0), -1)

    cv2.imshow("Finger tips", frame)      

    #quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()