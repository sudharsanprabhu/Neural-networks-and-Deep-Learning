# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:13:06 2020

@author: Prometheus
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

def relu(z):
    return np.maximum(0,z)

def sigmoidBackward(z):
    s=sigmoid(z)
    return s*(1-s)

def reluBackward(z):
    return (z>0)*1
        
def propagate(X,parameters):
    Z=[]
    A=[]
    
    
    A.append(X)
    Z.append(None)
    L=len(parameters)//2
    
    
    for i in range(1,L+1):
        z=np.dot(parameters['W'+str(i)],A[i-1])+parameters['b'+str(i)]
    
        if(i==L):
            a=sigmoid(z)

        else:
            a=relu(z)

        
        Z.append(z)
        A.append(a)
    
    
    return Z,A,A[-1]




def optimize(n,X,Y,rate):
   #Defining network
   layerDims=[X.shape[0],7,Y.shape[0]]
   L=len(layerDims)
   m=X.shape[1]
   
    
   #Initialize
   parameters={}
   costs=[]
   for i in range (1,L):
        parameters["W"+str(i)]=np.random.randn(layerDims[i],layerDims[i-1])*0.01
        parameters['b'+str(i)]=np.zeros((layerDims[i],1))
    
   for i in range (n): 
    
    #propagate
    Z,A,AL=propagate(X, parameters)
    
    #cost
    if(i%100==0):
      cost=np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))*(-1/m)
      print("Error after iteration "+str(i)+" : "+str(cost))
      costs.append(cost)
      
    #backwardPropagation
    grads={}
    
    #For Last layer
    ZL=Z[L-1]
    
    dAL=(-np.divide(Y,AL))+(np.divide(1-Y,1-AL))
    grads["dA"+str(L-1)]=dAL
    
    dZ=grads["dA"+str(L-1)]*sigmoidBackward(ZL)
    grads["dW"+str(L-1)]=np.dot(dZ,A[L-2].T)/m
    grads["db"+str(L-1)]=np.sum(dZ,axis=1,keepdims=True)/m
    grads["dA"+str(L-2)]=np.dot(parameters["W"+str(L-1)].T,dZ)
    
    #For other layers   
    for i in reversed(range(1,L-1)):
        dZ=grads["dA"+str(i)]*reluBackward(Z[i])
        grads["dW"+str(i)]=np.dot(dZ,A[i-1].T)/m
        grads["db"+str(i)]=np.sum(dZ,axis=1,keepdims=True)/m
        grads["dA"+str(i-1)]=np.dot(parameters["W"+str(i)].T,dZ)
        
    #Update parameters
    for i in range (1,L):
        parameters["W"+str(i)]=parameters["W"+str(i)]-(rate*grads["dW"+str(i)])
        parameters["b"+str(i)]=parameters["b"+str(i)]-(rate*grads["db"+str(i)])
        
   
   return parameters,costs


def predict(X,parameters):
    Z,A,AL=propagate(X, parameters)
    return (AL>0.5)*1



#Learn
parameters,costs=optimize(2500,trainSetX,trainSetY,0.0075)


plot.plot(costs)
plot.ylabel("Cost")
plot.xlabel("Iterations per hundereds")
plot.title("Deep NN Cost")
plot.show()

#Accuracy assessment
trainPrediction=predict(trainSetX,parameters)
testPrediction=predict(testSetX,parameters)



trainAccuracy=100-np.mean(np.abs(trainPrediction-trainSetY))*100
testAccuracy=100-np.mean(np.abs(testPrediction-testSetY))*100

print("Train Accuracy: "+str(trainAccuracy))
print("Test Accuracy: "+str(testAccuracy))
       

#%%

test=input("Test images(Y/N): ")
if(test=='Y'):
   for i in range(1,18):
        
    fname='i'+str(i)+".jpg"
    image=np.array(imageio.imread(fname))
    my_image=np.array(Image.fromarray(image).resize((numPx,numPx))).reshape(-1,1)
    my_image=my_image/255
    
    prediction=predict(my_image,parameters)
    
    print("Result "+str(i)+':'+classes[int(prediction)].decode("utf-8"))
    plot.imshow(image)
else:
    fname=input("Enter image path: ")
    image=np.array(imageio.imread(fname))
    my_image=np.array(Image.fromarray(image).resize((numPx,numPx))).reshape(-1,1)
    my_image=my_image/255
    
    prediction=predict(my_image,parameters)
    
    print("Result: "+classes[int(prediction)].decode("utf-8"))
    plot.imshow(image)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
