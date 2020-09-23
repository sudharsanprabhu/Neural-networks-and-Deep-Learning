# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:29:57 2020

@author: Sudharsan Prabhu
"""


import numpy as np
import h5py
import matplotlib.pyplot as plot
import imageio
from PIL import Image

numPx=0 
    
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
      
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    global numPx
    numPx=train_set_x_orig.shape[1]
    #Flatten
    trainX=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    testX=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    
    trainX=trainX/255
    testX=testX/255
    
    return trainX, train_set_y_orig, testX, test_set_y_orig, classes

trainSetX,trainSetY,testSetX,testSetY,classes=load_dataset()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def optimize(n,X,Y,rate):
    
    nX= X.shape[0]
    m = X.shape[1]
   
    costs=[]
    
    W=np.zeros((nX,1))
    b=0
    
    for i in range(n):
     z=np.dot(W.T,X)+b
     A=sigmoid(z)
     
     cost = (-1*np.sum((Y*np.log(A))+((1-Y)*np.log(1-A))))/m 
     
     dz=A-Y
     dw=np.dot(X,dz.T)/m
     db=np.sum(dz)/m
     
     W=W-rate*dw
     b=b-rate*db
    
     if i%100==0:
        costs.append(cost)
       
       
    return W,b,costs

def predict(W,b,X):
    w=W.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    YPrediction = np.zeros((1,X.shape[1]))
    
    for i in range(A.shape[1]):
        if(A[0,i]<=0.5):
            YPrediction[0,i]=0
        else:
            YPrediction[0,i]=1
        
    return YPrediction

W,b,costs=optimize(2000, trainSetX, trainSetY, 0.005)


plot.plot(costs)
plot.ylabel("Cost")
plot.xlabel("Iterations per hundereds")
plot.show()

trainYPrediction=predict(W,b,trainSetX)
testYPrediction=predict(W,b,testSetX)

trainAccuracy=100-np.mean(np.abs(trainYPrediction-trainSetY))*100
testAccuracy=100-np.mean(np.abs(testYPrediction-testSetY))*100

print("Train accuracy: "+str(trainAccuracy))
print("Test accuracy: "+str(testAccuracy))

#%%
test=input("Test images(Y/N: ")
if(test=='Y'):
   for i in range(1,18):
        
    fname='i'+str(i)+".jpg"
    image=np.array(imageio.imread(fname))
    my_image=np.array(Image.fromarray(image).resize((numPx,numPx))).reshape(-1,1)
    my_image=my_image/255
    
    prediction=predict(W,b,my_image)
    
    print("Result "+str(i)+':'+classes[int(prediction)].decode("utf-8"))
    plot.imshow(image)
else:
    fname=input("Enter image path: ")
    image=np.array(imageio.imread(fname))
    my_image=np.array(Image.fromarray(image).resize((numPx,numPx))).reshape(-1,1)
    my_image=my_image/255
    
    prediction=predict(W,b,my_image)
    
    print("Result: "+classes[int(prediction)].decode("utf-8"))
    plot.imshow(image)

    
     
        
        
    
    



















    
          
