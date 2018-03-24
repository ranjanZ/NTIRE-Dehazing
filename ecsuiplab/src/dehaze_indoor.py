import numpy as np
from scipy import sparse,ndimage
from scipy import misc
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.spatial import distance
from skimage.color import rgb2gray
import keras.backend as K
import sys
import os
import skimage.transform  as trans
from natsort import natsorted
from skimage.io import imread,imsave
from linear import *
import time





#os.environ['CUDA_VISIBLE_DEVICES'] ='-1'



### Initialization 
b_patch_X = 256
b_patch_Y = 256


try:
    hazy_image_dir=sys.argv[1]
    out_image_dir=sys.argv[2]
except:
    print"-----------------------------------------------------------------------------------------------------"
    print "Please provide hazy image directory  as 1st argument and  Outputdirectory as second argument"
    print "running in default mode....." 
    print"-----------------------------------------------------------------------------------------------------"
    hazy_image_dir= "../data/IndoorTestHazy/"
    out_image_dir= "../data/IndoorTestHazy_out/"


def read_image(path):
        image = imread(path,mode="RGB")
        image = image.astype('float32')/255
        return image

### Finding A and t from the model
def gt_filename(S):
    S=S.replace("hazy","GT")
    return S


def f(image,L1,L2,patch_Y,patch_X,model):

    h,w,c = image.shape
    Count=np.zeros((h,w))
    S=np.zeros((h,w))
    K_arr=np.zeros((h,w,c))
    T_arr=np.zeros((h,w))
    P=[]

    G=np.abs(np.gradient(rgb2gray(image),axis=[-1,-2]))
    G=G[1]+G[0]

    c=0
    for i in L1:
        for j in L2:
            patch = image[i : i+patch_Y,j:j+patch_X,:].copy()
            patch=trans.resize(patch,(b_patch_Y,b_patch_X,3))
            P.append(patch) 
            c=c+1

    P=np.array(P)        
    T, K = model.predict(P)

    c=0
    for i in L1:
        for j in L2:
            #print np.std(rgb2gray(P[c]).flatten())

            st1=np.mean(G[i : i+patch_Y,j:j+patch_X])

           
            t=trans.resize(T[c,:,:,0],(patch_Y,patch_X))
            k=trans.resize(K[c,:,:,:],(patch_Y,patch_X,3))
            K_arr[i : i+patch_Y,j:j+patch_X,:]=K_arr[i : i+patch_Y,j:j+patch_X,:]+k
            T_arr[i : i+patch_Y,j:j+patch_X]=T_arr[i : i+patch_Y,j:j+patch_X]+t
            
            Count[i : i+patch_Y,j:j+patch_X]=Count[i : i+patch_Y,j:j+patch_X]+1

            S[i : i+patch_Y,j:j+patch_X]=S[i : i+patch_Y,j:j+patch_X]+st1
            #Count[i : i+patch_Y,j:j+patch_X]=1
            c=c+1

                                                      
    T_arr[Count!=0] = T_arr[Count!=0]/Count[Count!=0]
    S[Count!=0] =S[Count!=0]/Count[Count!=0]
    Count=Count[:,:,np.newaxis]
    Count = np.tile(Count,(1,1,3))
    K_arr[Count!=0] = K_arr[Count!=0]/Count[Count!=0]

    return T_arr,K_arr,Count[:,:,0],S 
                



def new_dehaze(hazy_image_path,model):
       
    image_o=read_image(hazy_image_path)
    #image_o=open_img(hazy_image_path)

    
    k=(850.0/min(image_o.shape[0],image_o.shape[1]))
    image=trans.resize(image_o,(int(image_o.shape[0]*k),int(image_o.shape[1]*k),3))


    h,w,c = image.shape
    patch_Y=patch_X=128

    S=[patch_X*2,patch_X*3,patch_X*4]
    N=len(S)

    Count=np.zeros((N,h,w))
    STD=np.zeros((N,h,w))
    K_arr=np.zeros((N,h,w,c))
    T_arr=np.zeros((N,h,w))

    for c in range(0,N):
        patch_X=patch_Y=S[c]
        L1=range(0,h - patch_Y,patch_Y*2//4)
        L2=range(0, w - patch_X, patch_X*2//4)
        L1.append(h - patch_Y)
        L2.append(w - patch_X)  

        T,K,C,std1=f(image,L1,L2,patch_Y,patch_X,model)                          

        T_arr[c]=T
        K_arr[c]=K
        Count[c]=C
        STD[c]=std1

        c=c+1      
    

    std=0.0001
    S=np.mean(STD,axis=0)
    S[S>=std]=1
    S[S<std]=0


    s1=np.array([1.0,1,1])
    s2=np.array([1.0,1,1])

    #generate weight with sum 1
    s1=s1/np.sum(s1)
    s1=s1[:,np.newaxis,np.newaxis]    
    s2=s2/np.sum(s2)
    s2=s2[:,np.newaxis,np.newaxis]    
   
   
    #make a weight map 
    Count1=Count*s1
    Count2=Count*s2
    Count2=Count2[:,:,:,np.newaxis]

    
    #mul weight map
    K1=np.sum((K_arr*Count2),axis=0)
    T1=np.sum(T_arr*Count1,axis=0)

    #compute weight sum
    Count1_s=np.sum(Count1,axis=0)
    Count2_s=np.sum(Count2,axis=0)


    Count2_s=np.tile(Count2_s,(1,1,3))
    K1[Count2_s>0]=K1[Count2_s>0]/Count2_s[Count2_s>0]
    T1[Count1_s>0]=T1[Count1_s>0]/Count1_s[Count1_s>0]
    Count=Count[2]          #for linear interpolation only

   
     

    T1,K1=linear_smooth(image,T1,K1,S,sig1=0.1,sig2=0.1)




    T1=trans.resize(T1,(int(image_o.shape[0]),int(image_o.shape[1])))
    K1=trans.resize(K1,(int(image_o.shape[0]),int(image_o.shape[1]),3))

    return image_o,T1,K1,Count




### reconstructig the dehazed image from K,t and original hazy image
def dehaze(t,img,K):
    if(t is False):
        return False    

    t=t[:,:,np.newaxis]
    t = np.tile(t,(1,1,3))
    J=(img-K)/t

    J = J.clip(0,1)
    J[np.isnan(J)]=0

    return J
####



def dehaze_fast(hazy_image_path,model):

    img,T,K,S=new_dehaze(hazy_image_path,model)

    J = dehaze(T, img,K)
    return J,img,T,K


def evaluate(model_path):
    
    models=load_model(model_path)
    Time=0
    for  d in natsorted(os.listdir(hazy_image_dir)):
        start = time.time()        

        J,I,T,A=dehaze_fast(hazy_image_dir+d,models)
        end = time.time()
        t=(end-start)
        Time=Time+t
        print d+"  "+str(t)
        
        plt.imsave(out_image_dir+d,J,format="jpg")
        imsave(out_image_dir+d,J)
        imsave(out_image_dir+"A"+d,A)
        imsave(out_image_dir+"T"+d,T)

    print "average time :",Time/len(os.listdir(hazy_image_dir))



evaluate("../models/model_indoor.h5")


