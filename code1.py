import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img=cv2.imread('0.png')
img=np.float64(img)
blue,green,red=cv2.split(img)

blue[blue==0]=1
green[green==0]=1
red[red==0]=1

div=np.multiply(np.multiply(blue,green),red)**(1.0/3)

a=np.log1p((blue/div)-1)
b=np.log1p((green/div)-1)
c=np.log1p((red/div)-1)
a1 = np.atleast_3d(a)
b1 = np.atleast_3d(b)
c1 = np.atleast_3d(c)
rho= np.concatenate((c1,b1,a1),axis=2)

temp = rho[0,0,:]
temp = np.reshape(temp,(1,3))
U=[[1/math.sqrt(2),-1/math.sqrt(2),0],[1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
U=np.array(U)
X=np.dot(rho,U.T)
plt.scatter(X[:,:,0],X[:,:,1])
plt.show()
d1,d2,d3=img.shape
# #
# e_t=np.zeros((2,181))
# for j in range(181):
#     e_t[0][j]=math.cos(j*math.pi/180.0)
#     e_t[1][j]=math.sin(j*math.pi/180.0)
#
# Y=np.dot(X,e_t)
# nel=img.shape[0]*img.shape[1]
# bw=np.zeros((1,181))
#
#
# for i in range(181):
#     bw[0][i]=(3.5*np.std(Y[:,:,i]))*((nel)**(-1.0/3))
#
# entropy=[]
# for i in range(181):
#     temp=[]
#     comp1=np.mean(Y[:,:,i])-3*np.std(Y[:,:,i])
#     comp2=np.mean(Y[:,:,i])+3*np.std(Y[:,:,i])
#     for j in range(Y.shape[0]):
#         for k in range(Y.shape[1]):
#             if Y[j][k][i]>comp1 and Y[j][k][i]<comp2:
#                 temp.append(Y[j][k][i])
#     nbins=np.int(round((max(temp)-min(temp))/bw[0][i]))
#     (hist,waste)=np.histogram(temp,bins=nbins)
#     hist=filter(lambda var1: var1 != 0, hist)
#     hist1=np.array([float(var) for var in hist])
#     hist1=hist1/sum(hist1)
#     entropy.append(-1*sum(np.multiply(hist1,np.log2(hist1))))
#
# angle=entropy.index(min(entropy))
#
# e_t=np.array([math.cos(angle*math.pi/180.0),math.sin(angle*math.pi/180.0)])
# e=np.array([-1*math.sin(angle*math.pi/180.0),math.cos(angle*math.pi/180.0)])
#
# out_img=np.exp(np.dot(X,e_t))
# out_img = np.array(out_img)
# cv2.imwrite("img_out.png",out_img*255.0)
