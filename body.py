import mediapipe as mp
import cv2

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic


#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
#while cap.isOpened():
    #ret,frame=cap.read()
    #cv2.imshow('raw webcam feed',frame)

    #if cv2.waitKey(10) & 0xFF == ord('q'):
        #break
#cap.release()
#cv2.destroyAllWindows()
################################################

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
   while cap.isOpened():
     ret,frame=cap.read()
     image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
     results=holistic.process(image)
     print(results.face_landmarks)

     image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

     mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACE_CONNECTIONS)

    # mp_drawing.draw_landmarks(image,results.right_hand_handmarks,mp_holistic.HAND_CONNECTIONS)

     mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color=(240,0,0),thickness=2,circle_radius=2))

     mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)


     cv2.imshow('raw webcam feed',image)


     if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()