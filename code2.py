import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img=cv2.imread('11.png')
img3 = cv2.imread('11.png',0)
cv2.imwrite("img_extrinsic.png",img3)
edges = cv2.Canny(img3,200,200)
cv2.imwrite("img_extrinsic_edges.png",edges)
img=np.float64(img)
b, g, r = cv2.split(img)
b[b==0]=1
g[g==0]=1
r[r==0]=1
div=np.multiply(np.multiply(b,g),r)**(1.0/3.0)
a=np.log1p((b/div)-1)
b=np.log1p((g/div)-1)
c=np.log1p((r/div)-1)
vec = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        vec.append([c[i][j], b[i][j], a[i][j]])

vec = np.array(vec)

U = [[1/math.sqrt(2),-1/math.sqrt(2),0],[1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
U = np.array(U)
X = np.dot(vec,U.T)

ent = 1e18
ang = 0
for k in range(180):
    sum_ = 0
    temp = 0
    I = np.zeros((img.shape[0],img.shape[1]))
    ct = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            I[i][j] = np.abs(X[ct][0]*math.cos(k*math.pi/180.0) + X[ct][1]*math.sin(k*math.pi/180.0)) + 1e-50
            ct+=1
    cv2.imwrite("./out/img{}.png".format(k),I*255.0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum_ += I[i][j]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp += (I[i][j]/sum_) * math.log(I[i][j]/sum_)
    temp *= -1
    if(temp<ent):
        ent = temp
        ang = k
    print(k)

I = np.zeros((img.shape[0],img.shape[1]))
ct = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        I[i][j] = np.abs(X[ct][0]*math.cos(ang*math.pi/180.0) + X[ct][1]*math.sin(ang*math.pi/180.0))
        ct += 1

I = np.array(I)
print(ang)
cv2.imwrite("img_intrinsic.png",I*255.0)
img3 = cv2.imread("img_intrinsic.png",0)
edges = cv2.Canny(img3,200,200)
cv2.imwrite("img_intrinsic_edges.png",edges)

img1 = cv2.imread("img_intrinsic_edges.png",0)
img2 = cv2.imread("img_extrinsic_edges.png",0)

img3 = np.abs(img2-img1)
cv2.imwrite("shadow.png",img3)
