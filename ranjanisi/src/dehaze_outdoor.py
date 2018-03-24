import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.color import rgb2gray
import keras.backend as K
import sys,os
import skimage.transform  as trans
from gf import *
from natsort import natsorted
from skimage.io import imread,imsave
import time 


#os.environ['CUDA_VISIBLE_DEVICES'] ='-1'
# run cvpr.py  ../data/cvpr2018/IndoorValidationHazy/        ../data/cvpr2018/IndoorValidationHazy_out/


b_patch_X = 128
b_patch_Y = 128


try:
    hazy_image_dir=sys.argv[1]
    out_image_dir=sys.argv[2]
except:
    print "please give the patch for hazy image directory and output image directory"
    print"-----------------------------------------------------------------------------------------------------"
    print "Please provide hazy image directory  as 1st argument and  Outputdirectory as second argument"
    print "running in default mode....."
    print"-----------------------------------------------------------------------------------------------------"
    hazy_image_dir= "../data/OutdoorTestHazy/"
    out_image_dir= "../data/OutdoorTestHazy_out/"





def read_image(path):
        image = imread(path,mode="RGB")
        image = image.astype('float32')/255
        return image



def f(image,L1,L2,patch_Y,patch_X,model):

    h,w,c = image.shape
    Count=np.zeros((h,w))
    K_arr=np.zeros((h,w,c))
    T_arr=np.zeros((h,w))
    P=[]

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
                 
                t=trans.resize(T[c,:,:,0],(patch_Y,patch_X))
                k=trans.resize(K[c,:,:,:],(patch_Y,patch_X,3))
                K_arr[i : i+patch_Y,j:j+patch_X,:]=K_arr[i : i+patch_Y,j:j+patch_X,:]+k
                T_arr[i : i+patch_Y,j:j+patch_X]=T_arr[i : i+patch_Y,j:j+patch_X]+t
                

                Count[i : i+patch_Y,j:j+patch_X]=Count[i : i+patch_Y,j:j+patch_X]+1
                #Count[i : i+patch_Y,j:j+patch_X]=1
                c=c+1

                                                      
    T_arr[Count!=0] = T_arr[Count!=0]/Count[Count!=0]
    Count=Count[:,:,np.newaxis]
    Count = np.tile(Count,(1,1,3))
    K_arr[Count!=0] = K_arr[Count!=0]/Count[Count!=0]

    return T_arr,K_arr,Count[:,:,0] 
                



def new_dehaze(hazy_image_path,model):
       
    image_o=read_image(hazy_image_path)

    k=850.0/min(image_o.shape[0],image_o.shape[1])

    image=trans.resize(image_o,(int(image_o.shape[0]*k),int(image_o.shape[1]*k),3))



    h,w,c = image.shape
    patch_Y=patch_X=128

    S=[patch_X,patch_X*2,patch_X*3,patch_X*4]
    N=len(S)

    Count=np.zeros((N,h,w))
    K_arr=np.zeros((N,h,w,c))
    T_arr=np.zeros((N,h,w))

    for c in range(0,N):
        patch_X=patch_Y=S[c]
        L1=range(0,h - patch_Y,patch_Y*2//8)
        L2=range(0, w - patch_X, patch_X*2//8)
        L1.append(h - patch_Y)
        L2.append(w - patch_X)  

        T,K,C=f(image,L1,L2,patch_Y,patch_X,model)                          

        T_arr[c]=T
        K_arr[c]=K
        Count[c]=C
        c=c+1      



    s1=np.array([8.0,4,2,1])
    s2=np.array([8.0,4,2,1])

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
    T1=np.sum(T_arr*Count1,axis=0)
    K1=np.sum((K_arr*Count2),axis=0)

    #compute weight sum
    Count1_s=np.sum(Count1,axis=0)
    Count2_s=np.sum(Count2,axis=0)


    Count2_s=np.tile(Count2_s,(1,1,3))
    K1[Count2_s>0]=K1[Count2_s>0]/Count2_s[Count2_s>0]
    T1[Count1_s>0]=T1[Count1_s>0]/Count1_s[Count1_s>0]

   
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
    
    K=smooth_A_guided_filter(img,K,60,0.05)
    T=guided_filter(rgb2gray(img),T,40,0.05)

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
    
        #psnr=compute_psnr(out_iage_dir+d,gt_image_dir+gt_filename(d)) 
        #print psnr

    print "average time :",Time/len(os.listdir(hazy_image_dir))







evaluate('../models/mod_outdoor--.h5')   




