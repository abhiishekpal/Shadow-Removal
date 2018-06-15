import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction

def normal(img):
    mx = np.max(img)
    mi = np.min(img)
    img = (img-mi)/(mx-mi)
    return img

def main():
    # img = cv2.imread("h5_1.tif").astype(float)
    # img = normal(img)
    # img[img==0] = 0.01
    #
    # chrom = np.zeros((img.shape[0],img.shape[1],3),dtype = float)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         chrom[i][j][0] = img[i][j][0]/((img[i][j][0]*img[i][j][1]*img[i][j][2])**(1.0/3.0))
    #         chrom[i][j][1] = img[i][j][1]/((img[i][j][0]*img[i][j][1]*img[i][j][2])**(1.0/3.0))
    #         chrom[i][j][2] = img[i][j][2]/((img[i][j][0]*img[i][j][1]*img[i][j][2])**(1.0/3.0))
    #
    #
    # p3 = normal(np.log(chrom[:,:,0]))
    # p2 = normal(np.log(chrom[:,:,1]))
    # p1 = normal(np.log(chrom[:,:,2]))
    #
    # X1 = normal(p1*(1.0/(math.sqrt(2.0))) - p2*(1.0/(math.sqrt(2.0))))
    # X2 = normal(p1*(1.0/(math.sqrt(6.0))) + p2*(1.0/(math.sqrt(6.0))) - p3*(2.0/(math.sqrt(6.0))))
    #
    #
    # X = np.zeros((img.shape[0],img.shape[1],2))
    # X[:,:,0] = np.copy(X1)
    # X[:,:,1] = np.copy(X2)
    #
    # print(np.unique(X1))
    # for i in range(180):
    #     maxtheta = i
    #     InvariantImage2 = math.cos(maxtheta*np.pi/180.0)*X1 + math.sin(maxtheta*np.pi/180.0)*X2;
    #     img = normal(np.exp(InvariantImage2))*255.0
    #     ty =200
    #     # img[img>ty] = 2h5_150
    #     # img[img<ty]  = 0
    #     img = np.array(img)
    #     img = cv2.GaussianBlur(img,(11,11),0)
    #     cv2.imwrite("./new/h5_1-new_{}.tif".format(i),img)
    #         # print(i)
    #
    #
    # rot_mat = np.zeros((181,2))
    # for i in range(0,181):
    #     rot_mat[i][0] = math.cos(i*math.pi/180.0)
    #     rot_mat[i][1] = math.sin(i*math.pi/180.0)
    #
    # # print(X.shape)
    # rot_mat = np.array(rot_mat)
    # I = np.matmul(X, rot_mat.T)                                                 #images under various rotation
    #
    # total_energy = []
    # for i in range(181):
    #     # print(i)
    #     '''Removing Noise'''
    #     temp = []
    #     img = I[:,:,i]
    #     thres1 =np.mean(img)-3*np.std(img)
    #     thres2 =np.mean(img)+3*np.std(img)
    #     for j in range(img.shape[0]):
    #         for k in range(img.shape[1]):
    #             if img[j][k]>thres1 and img[j][k]<thres2:
    #                 temp.append(img[j][k])
    #
    #
    #     size = img.shape[0]*img.shape[1]
    #     range_ = max(temp) - min(temp)
    #     nbin = int(range_/(3.5 *np.std(img)*(size**(-1.0/3))))
    #     hist, waste = np.histogram(temp, bins = nbin*2)
    #     hist = hist.astype(float)
    #     hist = filter(lambda val: val != 0, hist)
    #     hist /= sum(hist)
    #     temp_entropy = -1*sum(np.multiply(hist,np.log2(hist)))
    #     total_energy.append(temp_entropy)
    #
    #
    # min_energy = min(total_energy)
    # min_angle = total_energy.index(min_energy)
    # InvariantImage2 = math.cos(min_angle*np.pi/180.0)*X1 + math.sin(min_angle*np.pi/180.0)*X2;
    # img = normal(np.exp(InvariantImage2))*255.0
    # cv2.imwrite("h5_1-new.tif",img)
    # # print(min_angle)
    # #
    # # e_t=np.array([-1*math.sin(min_angle*math.pi/180.0),math.cos(min_angle*math.pi/180.0)])
    # # P_theta = np.dot(e_t.T, e_t)
    # # X_theta = np.dot(X,P_theta)
    # #
    # # U=[[1/math.sqrt(2),-1/math.sqrt(2),0],
    # #   [1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
    # # U=np.array(U)
    # #
    # # rho_ti=np.dot(X_theta,U)
    # # c_ti=np.exp(rho_ti)
    # # sum_ti=np.sum(c_ti,axis=2)
    # # sum_ti=sum_ti.reshape(c_ti.shape[0],c_ti.shape[1],1)
    # # r_ti=c_ti/sum_ti
    # #
    # # r_ti2=255*r_ti
    # # r_ti2 = np.array(r_ti2)
    # # r_ti2 = cv2.GaussianBlur(r_ti2,(    25,25),0)
    # # cv2.imwrite("h5_1-intrinsic3D.tif",r_ti2)
    # # I1D = np.matmul(X,rot_mat[min_angle])
    # # img = normal(np.exp(I1D))*255.0
    # # img = cv2.GaussianBlur(img,(15,15),0)
    img_original = cv2.imread("h5_1.tif",0)
    img_original = cv2.medianBlur(img_original,71)
    img_original = cv2.GaussianBlur(img_original,(7,7),0)
    edges_org = cv2.Canny(img_original,13,13)
    cv2.imwrite("h5_1-org-edge.tif",edges_org)
    img_intrinsic = cv2.imread("h5_1-new.tif",0)
    img_intrinsic = cv2.medianBlur(img_intrinsic,71)
    img_intrinsic = cv2.GaussianBlur(img_intrinsic,(7,7),0)
    edges_int = cv2.Canny(img_intrinsic,13,13)
    cv2.imwrite("h5_1-new-edge.tif",edges_int)

    img_ty = abs(-1*edges_int+edges_org)
    print(img_ty.shape)
    cv2.imwrite("h5_1-boundary.tif",img_ty)
    #
    #
    #
    #
    #
    '''Adding a constant'''
    img = cv2.imread("h5_1.tif").astype(float)
    img_edges = cv2.imread("h5_1-boundary.tif",0).astype(float)
    # print(img_edges.shape)
    img = cv2.copyMakeBorder(img,25,25,25,25,0)
    img_edges = cv2.copyMakeBorder(img_edges,25,25,25,25,0)
    # print(img.shape)
    # print(img_edges.shape)
    ent_1 = []
    ent_2 = []
    ent_3 = []
    c1, c2, c3 = 100, 100, 100
    ct1 , ct2, ct3 = 1e19, 1e19, 1e19
    cost1, cost2, cost3 = [], [], []
    ad1, ad2, ad3 = [], [], []
    for i in range(img.shape[0]):
        print(i)
        for j in range(img.shape[1]):
            if(img_edges[i][j] == 255):
                # print(i,j)
                startx = i-5
                starty = j-5
                arr1 = np.zeros((5,5))
                arr2 = np.zeros((5,5))
                for k in range(-2,3):
                    for l in range(-2,3):
                        arr1[k][l] = img[startx+k][starty+l][0]

                startx = i+5
                starty = j+5
                for k in range(-2,3):
                    for l in range(-2,3):
                        arr2[k][l] = img[startx+k][starty+l][0]

                arr1 = np.array(arr1)
                arr2 = np.array(arr2)

                best = 1e15
                ad = 1000
                for k in range(255):
                    temp1 = np.copy(arr1)
                    temp2 = np.copy(arr2)
                    mx1 = np.max(temp1)
                    mx2 = np.max(temp2)
                    if(mx1+k<=255 and mx2+k<=255):
                        ans = temp2-temp1
                        ans += k
                        ans = np.sum(abs(ans))
                        if(ans<best):
                            best = ans
                            ad = k
                            # print(ad)

                ent_1.append(ad)
                if(best<ct1):
                    ct1 = best
                    c1 = ad
                    cost1.append(ct1)
                    ad1.append(c1)
                # print("*",best,ad)

            if(img_edges[i][j] == 255):
                startx = i-5
                starty = j-5
                arr1 = np.zeros((5,5))
                arr2 = np.zeros((5,5))
                for k in range(-2,3):
                    for l in range(-2,3):
                        arr1[k][l] = img[startx+k][starty+l][1]

                startx = i+5
                starty = j+5
                for k in range(-2,3):
                    for l in range(-2,3):
                        arr2[k][l] = img[startx+k][starty+l][1]

                arr1 = np.array(arr1)
                arr2 = np.array(arr2)

                best = 1e15
                ad = 1000
                for k in range(255):
                    temp1 = np.copy(arr1)
                    temp2 = np.copy(arr2)
                    mx1 = np.max(temp1)
                    mx2 = np.max(temp2)
                    if(mx1+k<=255 and mx2+k<=255):
                        ans = temp2-temp1
                        ans += k
                        ans = np.sum(abs(ans))
                        if(ans<best):
                            best = ans
                            ad = k

                ent_2.append(ad)
                if(best<ct2):
                    ct2 = best
                    c2 = ad
                    cost2.append(ct2)
                    ad2.append(c2)
                # print("**",best,ad)

            if(img_edges[i][j] == 255):
                startx = i-5
                starty = j-5
                arr1 = np.zeros((5,5))
                arr2 = np.zeros((5,5))
                for k in range(-2,3):
                    for l in range(-2,3):
                        arr1[k][l] = img[startx+k][starty+l][2]

                startx = i+5
                starty = j+5
                for k in range(-2,3):
                    for l in range(-2,3):
                        arr2[k][l] = img[startx+k][starty+l][2]

                arr1 = np.array(arr1)
                arr2 = np.array(arr2)

                best = 1e14
                ad = 1000
                for k in range(255):
                    temp1 = np.copy(arr1)
                    temp2 = np.copy(arr2)
                    mx1 = np.max(temp1)
                    mx2 = np.max(temp2)
                    if(mx1+k<=255 and mx2+k<=255):
                        ans = temp2-temp1
                        ans += k
                        ans = np.sum(abs(ans))
                        if(ans<best):
                            best = ans
                            ad = k

                ent_3.append(ad)
                if(best<ct3):
                    ct3 = best
                    c3 = ad
                    cost3.append(ct3)
                    ad3.append(c3)

    print(c1,c2,c3)
    ent_1 = np.unique(sorted(ent_1))
    ent_2 = np.unique(sorted(ent_2))
    ent_3 = np.unique(sorted(ent_3))

    ind1 = np.argsort(cost1)
    ind2 = np.argsort(cost2)
    ind3 = np.argsort(cost3)

    # print("*",np.unique(ent_1))
    # print("**",np.unique(ent_2))
    # print("***",np.unique(ent_3))

    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(2):
        sum_1 += ad1[ind1[i]]
        sum_2 += ad2[ind2[i]]
        sum_3 += ad3[ind3[i]]
        print(ad3[ind3[i]],"*"*10)

    sum_1 /= 2
    sum_2 /= 2
    sum_3 /= 2
    c1, c2, c3 = sum_1, sum_2, sum_3
    print(sum_1, sum_2, sum_3)
    l1 = len(ent_1)
    l2 = len(ent_2)
    l3 = len(ent_3)
    l11 = int(l1*0.01)
    # print(l11)
    for k in range(l11):
        sum_1 += ent_1[k]
    # print(sum_1)
    # sum_1 /= l11

    l22 = int(l2*0.01)
    for k in range(l22):
        sum_2 += ent_2[k]
    # print(sum_2)
    # sum_2 /= l22

    l33 = int(l3*0.01)
    for k in range(l33):
        sum_3 += ent_3[k]
    # print(sum_3)
    # sum_3 /= l33

    # print(sum_1,sum_2,sum_3)

    visited = np.zeros((img.shape[0],img.shape[1]))
    aa = []
    for j in range(7,img.shape[1]-7):
        op = 0
        for i in range(7,img.shape[0]-7):
            # print(visited[i][j])
            for k in range(i-3,i+4):
                for l in range(j-3,j+4):
                    if(img_edges[k][l]==255):
                        op+=5
                        break
                if(op==5):
                    break
            if(op>=5):
                if(visited[i][j] == 0 and img[i][j][2]<90 and img[i][j][2]<90 and img[i][j][1]<90 and img[i][j][0]<90):
                    img[i][j][0] += c1
                    img[i][j][1] += c2
                    img[i][j][2] += c3
                    visited[i][j] = 1




    img = np.array(img)
    cv2.imwrite("h51_regen.png",img)


if(__name__=="__main__"):
    main()
