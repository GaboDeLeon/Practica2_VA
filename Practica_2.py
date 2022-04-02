import cv2
import numpy as np
import matplotlib.pyplot as plt
control=0
while control<13:

 letra= input("Ingresa la G\n")

 if letra=="G":
     control=control+1

 if control==1:

     print("Suma")
     img1 = cv2.imread('imagen1_japon.jpg',0)
     img2 = cv2.imread('imagen2_montanas.jpg',0)
     resAW = cv2.addWeighted(img1,0.5,img2,0.5,0)
     cv2.imshow('Suma',resAW)
     cv2.imshow('Imagen 1',img1)
     cv2.imshow('Imagen 2',img2)
     cv2.moveWindow('Imagen 1', -1700,0)
     cv2.moveWindow('Imagen 2', -650,0)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==2:

     print("Resta")
     img1 = cv2.imread('imagen1_japon.jpg')
     img2 = cv2.imread('imagen2_montanas.jpg')
     resultado = cv2.subtract(img1,img2)
     print('img1[0,0]= ',img1[0,0])
     print('img2[0,0]= ',img2[0,0])
     print('resultado[0,0]= ',resultado[0,0])
     cv2.imshow('Resta',resultado)
     img1 = cv2.imread('imagen1_japon.jpg')
     cv2.imshow('Imagen 1',img1)
     img2 = cv2.imread('imagen2_montanas.jpg')
     cv2.imshow('Imagen 2',img2)
     cv2.moveWindow('Imagen 1', -1900,0)
     cv2.moveWindow('Imagen 2', -650,0)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==3:
     print("Division")
     img1 = cv2.imread('imagen1_japon.jpg')
     img2 = cv2.imread('imagen2_montanas.jpg')
     resultado = cv2.divide(img1,0.5,img2,0.5,1)
     cv2.imshow('Division',resultado)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


 elif control==4:
     print("Multiplicacion")
     img1 = cv2.imread('imagen1_japon.jpg')
     img2 = cv2.imread('imagen2_montanas.jpg')
     resultado_m = cv2.multiply(img1,0.5,img2,0.5,1)
     cv2.imshow('Multiplicacion',resultado_m)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==5:
     print("Logaritmo natural")
     img1 = cv2.imread('imagen1_japon.jpg')
     c=255/np.log(1+np.max(img1))
     log_image = c*(np.log(img1+1))
     log_image=np.array(log_image,dtype=np.uint8)
     plt.imshow(img1)
     plt.show()
     plt.imshow(log_image)
     plt.show()

 elif control==6:
     print("Derivada")
     img1 = cv2.imread('imagen1_japon.jpg')
     imgx = cv2.Sobel(img1,cv2.CV_16S,1,0,ksize=3)
     imgy = cv2.Sobel(img1,cv2.CV_16S,0,1,ksize=3)
     #   uint8
     imgx_uint8 = cv2.convertScaleAbs(imgx)
     imgy_uint8 = cv2.convertScaleAbs(imgy)
     # x, combinación de dirección y
     img = cv2.addWeighted(imgx_uint8,0.5,imgy_uint8,0.5,0)
     cv2.imshow('sobelimg',img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 

 elif control==7:
     print("Conjuncion")
     img1 = cv2.imread('imagen1_japon.jpg')
     img2 = cv2.imread('imagen2_montanas.jpg')
     resultado = cv2.bitwise_and (img1,img2)
     cv2.imshow('Conjuncion',resultado)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


 elif control==8:
     print("Disyuncion")
     img1 = cv2.imread('imagen1_japon.jpg')
     img2 = cv2.imread('imagen2_montanas.jpg')
     resultado = cv2.bitwise_or (img1, img2)
     cv2.imshow('Disyuncion',resultado)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==9:
     print("Negacion")
     img1 = cv2.imread('imagen1_japon.jpg')
     resultado = cv2.bitwise_not (img1)
     cv2.imshow('Negacion',resultado)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==10:
     print("Traslacion")
     img = cv2.imread('imagen1_japon.jpg',0)
     rows,cols = img.shape
     M = np.float32([[1,0,210],[0,1,20]])
     dst = cv2.warpAffine(img,M,(cols,rows))
     cv2.imshow('Traslacion',dst)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==11:
     print("Escalado")
     import cv2
     img = cv2.imread('imagen1_japon.jpg')
     newImg = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
     cv2.imshow('Escalado', newImg)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 elif control==12:
     print("Rotacion")
     img = cv2.imread('imagen1_japon.jpg',0)
     rows,cols = img.shape
     M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
     dst = cv2.warpAffine(img,M,(cols,rows))
     cv2.imshow('Rotacion', dst)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

 else:
     print("Traslacion A fin")
     img = cv2.imread('imagen1_japon.jpg')
     rows,cols,ch = img.shape
     pts1 = np.float32([[100,400],[400,100],[100,100]])
     pts2 = np.float32([[50,300],[400,200],[80,150]])
     M = cv2.getAffineTransform(pts1,pts2)
     dst = cv2.warpAffine(img,M,(cols,rows))
     plt.subplot(121),plt.imshow(img),plt.title('Input')
     plt.subplot(122),plt.imshow(dst),plt.title('Output')
     plt.show()
