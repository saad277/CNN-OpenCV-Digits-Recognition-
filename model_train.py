import numpy as np 
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

path="DataSet"

#get dataset
myList=os.listdir(path)

print("Total classes : ",myList)

images=[]
classNo=[]

noOfClasses=len(myList)

print(noOfClasses)

for x in range(0,noOfClasses):                      
    
    myPicList=os.listdir(path+"/"+str(x))           #getting all folders 

    for y in myPicList:                             #getting individual images in folder 
 
        currentImage=cv2.imread(path+"/"+str(x)+"/"+y)

        currentImage=cv2.resize(currentImage,(32,32))

        images.append(currentImage)
        classNo.append(x)


print("Length of images ",len(images))
print("Length of classNo :",len(classNo));

#Converting to numpy arrays

images=np.array(images);
classNo=np.array(classNo);


print("shape of images : ",images.shape);
print("shape of classNo : ",classNo.shape)

#Splitting Data

x_train, x_test, y_train , y_test =train_test_split(images,classNo,test_size=0.2)       #20% testing and 80% training     

x_train, x_valid, y_train, y_valid =train_test_split(x_train,y_train,test_size=0.2)

print("train shape : ",x_train.shape)
print("test shape  : ",x_test.shape)


print("valid shape :",x_valid.shape)
print("valid test  :",y_valid.shape)

numOfSamples=[]

#getting labels from y_train seperately

for x in range(0,noOfClasses):
    #print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))

print(numOfSamples)  #how many indexes have the numbers


def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    img=cv2.equalizeHist(img);
    img=img/255
    return img

##img=preProcessing(x_train[30]);
##img=cv2.resize(img,(300,300))
##cv2.imshow("Pre-Processed Image : ",img)
##cv2.waitKey(0)

#using map function to pre-process each image

x_train=np.array(list(map(preProcessing,x_train)))
x_test=np.array(list(map(preProcessing,x_test)))
x_valid=np.array(list(map(preProcessing,x_valid)))

#print(x_train.shape)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_valid=x_valid.reshape(x_valid.shape[0],x_valid.shape[1],x_valid.shape[2],1)

#print(x_train.shape)
#print(x_test.shape)
#print(x_valid.shape)

dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)

dataGen.fit(x_train)


#Hot encoding
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,noOfClasses)
y_valid=to_categorical(y_valid,noOfClasses)

def myModel():


    model=Sequential()

    model.add(Conv2D(filters=60,kernel_size=(5,5),input_shape=(32,32,1)))

    model.add(Activation("relu"))

    model.add(Conv2D(filters=60,kernel_size=(5,5)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=30,kernel_size=(3,3)))
    
    model.add(Activation("relu"))

    model.add(Conv2D(filters=30,kernel_size=(3,3)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(rate=0.5))

    model.add(Flatten())

    model.add(Dense(units=500))

    model.add(Activation("relu"))

    model.add(Dropout(rate=0.5))

    model.add(Dense(10));

    model.add(Activation("softmax"))

    

   

    model.compile(optimizer=Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])

    return model


model=myModel()

print(model.summary())

history=model.fit_generator(dataGen.flow(x_train,y_train,batch_size=50),
                    steps_per_epoch=1200,
                    epochs=4,
                    validation_data=(x_valid,y_valid))


model.save(filepath=r"C:\Users\Ammad\Desktop\Digits Recognizer\model_save.h5",overwrite=True);


