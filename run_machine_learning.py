import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawning = mp.solutions.drawing_utils
mp_drawning_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_dict = {0:'A', 1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L', 12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
temp = '*'
word = ''
apple = 0
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape    

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawning.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawning_styles.get_default_hand_landmarks_style(),
                mp_drawning_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
        x1 = int(min(x_)*W) -10
        y1 = int(min(y_)*H) -10

        x2 = int(max(x_)*W) -10
        y2 = int(max(y_)*H) -10
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = label_dict[int(prediction[0])]

        if (predicted_character != temp):
             word =  word + predicted_character
             temp = predicted_character
        if (len(word) > 20):
            word = "restart"
            apple = apple +1
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
        cv2.putText(frame, predicted_character, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, word, (x1+50, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        if (apple <= 5 and apple >0):# increase apple count to make restart stay on screen longer and stop the appending of strings
            apple +=1
        if(apple >=5):
            word = " "
            apple = 0 

    cv2.imshow('frame',frame)
    cv2.waitKey(25)
cap.release()
cv2.destroyAllWindows()