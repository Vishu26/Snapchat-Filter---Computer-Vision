from cv2 import *
import scipy.misc
import numpy as np
import random

def sp_noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = [0, 0, 0]
            elif rdn > thres:
                output[i][j] = [255, 255, 255]
            else:
                output[i][j] = image[i][j]
    return output

flag = 2
face_cascade = CascadeClassifier('face.xml')
love = imread('filter.jpg')
gr = cvtColor(love, COLOR_BGR2GRAY)
ret, thresh = threshold(gr, 220, 255, THRESH_BINARY)
no = bitwise_not(thresh)
fi = bitwise_and(love, love, mask=no)

love2 = imread('love.jpg')
gr2 = cvtColor(love2, COLOR_BGR2GRAY)
ret, thresh2 = threshold(gr2, 220, 255, THRESH_BINARY)
no2 = bitwise_not(thresh2)
fi2 = bitwise_and(love2, love2, mask=no2)

cap = VideoCapture(0)

while 1:
    ret, img = cap.read()

    gray = cvtColor(img, COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #noi = sp_noise(img, 0.08)

    for (x,y,w,h) in faces:
        try:
            if flag == 1:
                t = scipy.misc.imresize(thresh2, (h, w))
                f = scipy.misc.imresize(fi2, (h, w))
                roi = bitwise_and(img[y:y + h, x:x + w], img[y:y + h, x:x + w], mask=t)
                img[y:y + h , x:x + w] = bitwise_or(roi, f)
            elif flag==2:
                t = scipy.misc.imresize(thresh, (h+40, w+40))
                f = scipy.misc.imresize(fi, (h+40, w+40))
                roi = bitwise_and(img[y-90:y+h-50, x-20:x+w+20], img[y-90:y+h-50, x-20:x+w+20], mask=t)
                img[y-90:y + h-50, x-20:x + w+20] = bitwise_or(roi, f)

        except:
            pass


    imshow('python',img)


    k = waitKey(30) & 0xff
    if k ==27:
        break
    if k == 13:
        flag = 1 if flag==2 else 2
    
cap.release()
destroyAllWindows()
