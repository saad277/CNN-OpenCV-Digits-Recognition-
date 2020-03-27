import numpy as np 
import cv2
from keras.models import load_model


model=load_model("model_save.h5");


width = 640
height = 480

cap=cv2.VideoCapture(0);

cap.set(3,width)
cap.set(4,height)

threshold=0.65

def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    img=cv2.equalizeHist(img);
    img=img/255
    return img


while True:
    ret,frame=cap.read();
    img=np.asarray(frame);
    img=cv2.resize(img,(32,32))
    img=preProcessing(img)
    
    img=img.reshape(1,32,32,1)
    #predict
    classIndex=int(model.predict_classes(img))
    print(classIndex)
    predictions=model.predict(img)
    probVal=np.amax(predictions)
    print(classIndex," --- >",probVal)

    if probVal>threshold:
        cv2.putText(frame,str(classIndex),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),4)

        cv2.putText(frame,"Detected",
                    (50,90),cv2.FONT_HERSHEY_COMPLEX,
                    1,(255,0,0),4)
        

    cv2.imshow("Processed Image ",frame )
    


    if cv2.waitKey(1) &0xFF==ord("q"):
        break;
