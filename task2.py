from task1 import Gaussian_Morph,bilateral_Morph,showWait,drawcntMap,getMostContour,mergeBoundingBoxes,filterWithWhite,train_for_Arrows,train_for_Digits
import numpy as np 
import random
import os
import sys
import glob
import cv2
import argparse
task_num = 'task2'
BASE_PATH = os.path.abspath('')
# loading models
# train the algorithm for digits
cell_size = (5, 5)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins
hog = cv2.HOGDescriptor(_winSize=(25 // cell_size[1] * cell_size[1],
                                    25 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
n_cells = (25 // cell_size[0], 25 // cell_size[1])
digit_svm = cv2.ml.SVM_create()
digit_knn = cv2.ml.KNearest_create()
print('????')
if os.path.exists('digit-svm.xml') and os.path.exists('digit-data.npy') and os.path.exists('digit-labels.npy'):
    digit_svm = digit_svm.load('digit-svm.xml')
    digit_data = np.load('digit-data.npy')
    digit_labels = np.load('digit-labels.npy')
    digit_knn.train(digit_data, cv2.ml.ROW_SAMPLE, digit_labels)
else:
    digit_svm,digit_knn = train_for_Digits(digit_svm,digit_knn,cell_size,block_size,nbins,n_cells)
# and for arrows
arrow_svm = cv2.ml.SVM_create()
arrow_knn = cv2.ml.KNearest_create()
if os.path.exists('arrow-svm.xml') and os.path.exists('arrow-data.npy') and os.path.exists('arrow-labels.npy'):
    arrow_svm = digit_svm.load('arrow-svm.xml')
    arrow_data = np.load('arrow-data.npy')
    arrow_labels = np.load('arrow-labels.npy')
    arrow_knn.train(arrow_data, cv2.ml.ROW_SAMPLE, arrow_labels)
else:
    arrow_svm,arrow_knn = train_for_Arrows(arrow_svm,arrow_knn,cell_size,block_size,nbins,n_cells)
imgpth = os.path.join(os.path.join(os.path.join('home','student'),'test'),'task2')
imgpath = os.path.join(BASE_PATH,imgpth)
ds_text = []

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-show', "--showFilters", help='show each filter applied on image',type=bool)
    parser.add_argument('-i', "--imgFolder", help='Image folder name')
    parser.add_argument('-r','--resultsOnly',help='Show results only',type=bool)
    try:
        args = parser.parse_args()
    except argparse.ArgumentError() as e:
        print(e)
    if args.imgFolder is not None:
        imgpath = args.imgFolder
    for imgpth in glob.glob(os.path.join(imgpath,'*.jpg')):
        base = os.path.basename(imgpth)
        file_name = os.path.splitext(base)[0]
        img = cv2.imread(imgpth)
        img = cv2.resize(img, (540, 720))  
        # Filters for digit recognition
        bilateral_filter = bilateral_Morph(img,7,20,20,500,700)
        gaussian_filter = Gaussian_Morph(img,9,3,250,300)
        gaussian_mediocre_filter = Gaussian_Morph(img,7,5,150,300)
        gaussian_hard_filter = Gaussian_Morph(img,9,3,500,800)
        white_filter = filterWithWhite(img,70)
        print('why not')
        # using this much filters because directional signages are 
        # hard to filter out from environment where as building signages
        # are easy to detect because they're most of the time on a plain wall
        filterArr = [gaussian_filter,white_filter,gaussian_hard_filter,gaussian_mediocre_filter,bilateral_filter]
        if args.showFilters and not args.resultsOnly:
            for flt in filterArr:
                showWait(flt,'filters used for digits')
        
        # Height threshold optimization for individual photos
        # Creates an array from 18 to 10 with decrementing by 2
        # checks how many contours have found, if nothing changes on contour
        # amount in consequent filters then take this as the threshold.
        dig_contours = [1]
        h_thresh = np.arange(18,10,-2)
        dig_hist = []
        for h_th in h_thresh:
            dig_contours,dig_labels,dig_img = getMostContour(img,digit_svm,digit_knn,filterArr,digits=True,wThresh=5,hThresh=h_th) # 9 -21
            dig_hist.append(len(dig_contours))
        for i in range(len(dig_hist)-1):
            if dig_hist[i+1] == dig_hist[i]:
                opt_H = h_thresh[i]
        print(opt_H)
        # End of height optimization
        # Filters for arrow recognition
        gaussian_filter = Gaussian_Morph(img,7,5,150,300)
        gaussian_hard_filter = Gaussian_Morph(img,11,5,300,500)
        gaussian_mediocre_filter = Gaussian_Morph(img,8,5,150,300)
        filterArr = [gaussian_filter,gaussian_hard_filter,gaussian_mediocre_filter]
        if args.showFilters and not args.resultsOnly:
            for flt in filterArr:
                showWait(flt,'filters used for arrows')
        #arrow_contours,arrow_labels,arrow_img = getMostContour(img,arrow_svm,arrow_knn,filterArr,digits=False,wThresh=3,hThresh=2) # 12 - 8
        counter_list = []
        # Merge arrow contours with initially merged digit contours 
        # Take the one which has highest number of merged bounding boxes
        for flt in filterArr:
            arrow_contours,arrow_labels,arrow_img = getMostContour(img,arrow_svm,arrow_knn,[flt],digits=False,wThresh=3,hThresh=2) # 12 - 8
            bbList,result,resulted_boxes,ds_labels,counter = mergeBoundingBoxes(img.copy(),dig_contours,dig_labels,arrow_contours,arrow_labels=arrow_labels,write=True,task_num='task2')
            counter_list.append((counter,(result,ds_labels)))
        counter_list = np.asarray(counter_list)
        result,ds_label = counter_list[np.argmax(counter_list[:,0]),1]
        # append resulted labels to a list
        if len(ds_labels)>0:
            lbls = ''
            for lbl in ds_labels:
                lbls += lbl+', '
            ds_text.append(file_name+': '+str(lbls)+'\n')
        if args.resultsOnly:
            showWait(result,'result')
    pth = os.path.join(os.path.join(os.path.abspath(''),'output'),str(task_num))
    if os.path.exists(os.path.join(pth,"DirectionalAnswer.txt")):
        os.remove('DirectionalAnswer.txt')
    ds_file = open('DirectionalAnswer.txt','w')
    ds_file.writelines(ds_text)
    ds_file.close()