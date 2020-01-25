import cv2
import numpy as np
import sys,getopt,glob,random,os
import argparse

BASE_PATH = os.path.abspath('')
task_num = 'task1'

def showWait(img, title):
    """
    A function for easy implementation of image showing.
    img: image to show
    title: title of the image window
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bb_intersection_over_union(boxA, boxB):
    # reference: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


def Gaussian_Morph(img,ksize,blur_ksize,canny1,canny2,morph=True):
    # Gaussian blur and canny
    # Standard working mask
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)     
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    # best kernel found by trial,      
    if morph:
        kernel = np.ones((ksize, ksize), np.uint8)     
        # I wanted to use morph tophat, since we have a lot of noise in the pictures
        blur = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)        
    edges = cv2.Canny(blur, canny1, canny2)
    return edges

def bilateral_Morph(img,ksize,blur1,blur2,canny1,canny2,morph=True):
    # Bilateral Blurring for keeping the edges sharp
    # Works when some contours are merged
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 3, blur1, blur2)
    if morph:
        kernel = np.ones((ksize, ksize), np.uint8)
        blur = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    edges = cv2.Canny(blur, canny1,canny2)
    return edges

def filterWithWhite(image,sens):
    """
    Filters the image according to white color presence.
    I wanted to apply this filter since we have the knowledge that all 
    signages will be white colored digits on either yellow or black 
    background.

    image: Image to be filtered
    sens: Sensitivity value of white color filtering.
    For further information see the opencv documentation page
    link here
    """
    # convert image to hsv color space 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = sens
    # define ranges of white color in HSV colorspace according to sensitivity
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])
    # create a mask if the color value in pixel, lower<pixel<upper == 1 else 0 
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # apply bitwise and to the image
    summ = cv2.bitwise_and(image.copy(), image.copy(), mask = mask)
    # turn image to gray for thresholding
    summ = cv2.cvtColor(summ,cv2.COLOR_BGR2GRAY)
    # blur for noise reducing 
    summ = cv2.GaussianBlur(summ, (3, 3), 0)
    # threshold the pixel values higher than 70
    summ = cv2.threshold(summ,70,255,cv2.THRESH_BINARY)[1]
    return summ

def drawcntMap(orgimg,filteredimg,wThresh,hThresh):
    """
    Function to find and draw contours then draw bounding boxes according to contours.

    filteredimg: Photo processed by oen of the two filters on the top.
    """
    _, contour, _ = cv2.findContours(filteredimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = cv2.drawContours(orgimg.copy(), contour, -1, (0, 0, 255), 2)  # To draw filtered contours on original image

    digitCnts = []  # contours to be surrounded by bounding boxes

    for c in contour:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= wThresh  and h >= hThresh and w*h <20000:  # Length filters to reduce noise
            cv2.rectangle(cnt,(x,y),(x+w,y+h),[0,255,0],2)
            digitCnts.append(c)

    return cnt, digitCnts

def secondElimination(dig_cnt,labels):
    resulted_cnt = []
    res_labels = []
    marked_arr = np.zeros(len(dig_cnt))
    inside_cnt = []
    i = 0
    while marked_arr[marked_arr==0].size>0:
        if marked_arr[i] == 1:
            pass
        else:
            bb1 = cv2.boundingRect(dig_cnt[i])
        
            marked_arr[i] = 1
        
            for j,cnt2 in enumerate(dig_cnt):
                if j==i or marked_arr[j] == 1:
                    pass
                else:
                    bb2 = cv2.boundingRect(cnt2)
                    iou = bb_intersection_over_union((bb1[0],bb1[1],bb1[0]+bb1[2],bb1[1]+bb1[3]),(bb2[0],bb2[1],bb2[0]+bb2[2],bb2[1]+bb2[3]))
                    if iou>0.2:
                        marked_arr[j] = 1
                        inside_cnt.append(j)
        i+=1
    for i,cnt in enumerate(dig_cnt):
        if i not in inside_cnt:
            resulted_cnt.append(cnt)
            res_labels.append(labels[i])
    return resulted_cnt,res_labels
        

# ML models initalization for digit recognition
digit_svm = cv2.ml.SVM_create()
digit_knn = cv2.ml.KNearest_create()
cell_size = (5, 5)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

# this approach is based on pixelou's answer in this link : https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python

# winSize is the size of the image cropped to an multiple of the cell size
# HOG ( Histogram of Oriented Gradients) method will be used instead of 
# raw pixel values for training, since it provides more useful information
# about images. Also increases accuracy a great amount.
hog = cv2.HOGDescriptor(_winSize=(25 // cell_size[1] * cell_size[1],
                                    25 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

n_cells = (25 // cell_size[0], 25 // cell_size[1])
def train_for_Digits(svm,knn,cell_size,block_size,nbins,n_cells,save=True):
    # ML Algorithm training phase, 
    sort_digs = os.path.join(BASE_PATH,'Sorted Digits\\')
    allDigits = []
    # loop through every digit image and add them to database
    # digit images were extracted by our first algorithm, then classified by hand,
    # data size is increased with augmentation methods
    for i in range(12):
        path = sort_digs+str(i)
        digit_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        i=0
        for dig in digit_paths:
            image = cv2.imread(dig)
            # resize image so we can use hog decsriptor
            image = cv2.resize(image,(25,25))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # reshape the hog feature outcome so that it would be
            # (cell_size,cell_size,block_size,block_size,Features)
            hog_feats = hog.compute(image)\
                .reshape(n_cells[1] - block_size[1] + 1,
                            n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1], nbins) \
                .transpose((1, 0, 2, 3, 4))
            # Flatten the array and append to allDigits
            allDigits.append(hog_feats.flatten())
    # Set training data and labels respectively, then train classifiers
    allDigits = np.array(allDigits, dtype=np.float32)
    Train_digits = allDigits
    k = np.arange(12)
    Train_labels = np.repeat(k, 200)[:,np.newaxis]
    svm.setType(cv2.ml.SVM_C_SVC)
    # Linear kernel is the most successful one in our case that works best 
    # with our filtering method, since in multi-dimensional space a discrimaniton
    # with a linear line between numbers in the same font face would make sense.
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # standard values, see opencv documentation for SVM...
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6))
    svm.train(Train_digits, cv2.ml.ROW_SAMPLE, Train_labels)
    knn.train(Train_digits, cv2.ml.ROW_SAMPLE, Train_labels)
    if save:
        np.save('digit-data.npy',allDigits)
        np.save('digit-labels.npy',Train_labels)
        svm.save('digit-svm.xml')
    return svm,knn
# trainin function for directional arrow recognition
def train_for_Arrows(svm,knn,cell_size,block_size,nbins,n_cells,save=True):
    # ML Algorithm training phase,
    sort_digs = os.path.join(BASE_PATH,'Sorted DA\\')
    allDigits = []
    # loop through every digit image and add them to database
    # digit images were extracted by our first algorithm, then classified by hand,
    # data size is increased with augmentation methods
    for i in range(3):
        path = sort_digs+str(i)
        digit_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for dig in digit_paths:
            image = cv2.imread(dig)
            # resize image so we can use hog decsriptor
            image = cv2.resize(image,(25,25))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # reshape the hog feature outcome so that it would be
            # (cell_size,cell_size,block_size,block_size,Features)
            hog_feats = hog.compute(image)\
                .reshape(n_cells[1] - block_size[1] + 1,
                            n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1], nbins) \
                .transpose((1, 0, 2, 3, 4))
            # Flatten the array and append to allDigits
            allDigits.append(hog_feats.flatten())
    # Set training data and labels respectively, then train classifiers
    allDigits = np.array(allDigits, dtype=np.float32)
    Train_digits = allDigits
    k = np.arange(3)
    Train_labels = np.repeat(k, 200)
    svm.setType(cv2.ml.SVM_C_SVC)
    # Linear kernel is the most successful one in our case that works best
    # with our filtering method, since in multi-dimensional a discrimaniton
    # with a linear line between numbers in the same font face would make sense.
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # standard values, see opencv documentation for SVM...
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6))
    svm.train(Train_digits, cv2.ml.ROW_SAMPLE, Train_labels)
    knn.train(Train_digits, cv2.ml.ROW_SAMPLE, Train_labels)
    if save:
        np.save('arrow-data.npy',allDigits)
        np.save('arrow-labels.npy',Train_labels)
        svm.save('arrow-svm.xml')
    return svm,knn
def cropNwriteBBs(img, digitCnts,writeToFile=False,folderName='outcome',cropW=20,cropH=30):
    """
    This function draws bounding boxes around given contours,
    then crops the resulted bounding box and yields it.
    digitCnts: Contour parameter.
    img: Original image that crops will be taken from.
    """
    d = 0
    for contour in digitCnts:
        # https://stackoverflow.com/questions/50331025/how-to-crop-a-bounding-box-out-of-an-image
        (x, y, w, h) = cv2.boundingRect(contour)

        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

        img1 = img.copy()
        img2 = img.copy()
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped_image = img2[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        cropped_image = cv2.resize(cropped_image, (cropW, cropH))
        filename = os.path.join(os.path.join(BASE_PATH,folderName),'output_%d'%d+'.jpg')
        d += 1
        if writeToFile:
            cv2.imwrite(filename, cropped_image)
            cv2.waitKey(0)
        yield cropped_image,(x,y,w,h),contour

def getMostContour(img,svm,knn,filterArr,digits,wThresh,hThresh):
    """
    Function to choose whether filter function to use.
    This is used as a scoring function to determine which filter detects more digits,
    but since our machine learning algorithm is not %100 perfect all the time, this scoring 
    method is not successfull all the time too. So instead of choosing one from multiple filters
    I decided to take only the best filtering method.
    Simply, compare number of the bounding boxes in each filters to get which one is more successful.

    img: Image to detect digits.
    svm: SVM Classifier Instance to train
    knn : KNN Classifier Instance to train
    filterArr: Array of filters
    digits: Boolean, set True if training for digit recognition, False for arrow recognition
    """
    # append the filter to filter array, this approach is used in case of 
    # multiple filter methods would be used.
    counts = []
    # iterare through every filter
    for flt in filterArr:
        # copy the image so we don't draw on same image
        flt_img = img.copy()
        last_img = img.copy()
        flt_contour,cntfound_fltr = drawcntMap(img.copy(),flt,wThresh,hThresh) 
        if not digits:
            flt_contour,cntfound_fltr = drawcntMap(img.copy(),flt,wThresh,hThresh)
        flt_contour_map = []
        labels = []
        for crop,(x,y,w,h),contour in cropNwriteBBs(img,cntfound_fltr):
            #crop = np.array(crop,dtype='float32')
            crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop,(25,25))
            # winSize is the size of the image cropped to an multiple of the cell size
            hog_fts = hog.compute(crop)\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               .transpose((1, 0, 2, 3, 4))
            hog_fts = np.resize(hog_fts.flatten(),(1,576))
            # make the resulted crop same type with the trained values
            hog_fts.dtype = 'float32'
            # get predicted labels
            label_svm=svm.predict(hog_fts)[1]
            label_knn = knn.findNearest(hog_fts,k=5)[1]
            # label 10 is considered as 'not digit' or 'thrash'
            # so if predicted label is not 10, draw the bounding box
            if digits:
                if(label_svm!=10 and label_knn != 10 and label_svm!=11 and label_knn != 11):
                    flt_contour_map.append(contour)
                    labels.append(str(label_knn[0])[1])
            else:
                if(label_svm!=2 and label_knn != 2):
                    flt_contour_map.append(contour)
                    labels.append(str(label_knn[0])[1])
                    #cv2.putText(flt_img,str(label_knn[0])[1],(x,y),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.8,color=(0,0,255))
                #cv2.putText(flt_img,str(label_knn[0])[1],(x,y),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.8,color=(0,0,255))
        last_cnt,last_labels = secondElimination(flt_contour_map,labels)
        for cnt in last_cnt:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(flt_img,(x,y),(x+w,y+h),[0,255,0],2)
        #showWait(flt_img,'fltres')
        _,xx,res_boxes,_,_ = mergeBoundingBoxes(flt_img,last_cnt,last_labels)
        cnt = len(res_boxes)
        counts.append([cnt,flt_img,last_cnt,last_labels])
        # append resulted image and contours to an array
    counts = np.asarray(counts)
    # get the resulted image which contain more digits (bounding boxes)
    tmp = counts[:,0]
    resulted_img = counts[np.argmax(tmp),1]
    result_labels = counts[np.argmax(tmp),3]
    resulted_contour = counts[np.argmax(tmp),2]
    return resulted_contour,result_labels,resulted_img


def mergeBoundingBoxes(img,cnts,label_svm,arrow_cnt=None,arrow_labels=None,write=False,task_num='task1'):
    temp_img = img.copy()
    labels = []
    arrow_counter = 0
    try:
        # create a list of bounding boxes
        resulted_boxes = []
        bbList = np.zeros((len(cnts),8))
        if arrow_cnt is not None:
            arrow_bbs = np.zeros((len(arrow_cnt),4))
            for j,cnt in enumerate(arrow_cnt):
                x,y,w,h = cv2.boundingRect(cnt)
                arrow_bbs[j,:4] = x,y,w,h
        # initialize bouding box list with x,y,w,h,center X,center Y, label of bb, Group number
        # initialize group numbers as all 0s at first.
        for i,cnt in enumerate(cnts):
            x,y,w,h = cv2.boundingRect(cnt)
            bbList[i,:4] = x,y,w,h
            cX = int(x+w*0.5)
            cY = int(y+h*0.5)
            bbList[i,4] = cX
            bbList[i,5] = cY
            bbList[i,6] = label_svm[i]
        i=0
        c = 1
        # iterate till there aren't any classified bbs.
        while bbList[bbList[:,-1]==0,-1].size>0:
            arrow_lbl = []
            bb1 = bbList[i]
            count = 0
            # if current bb is classified as in group or as a 'loner'
            if bbList[i,-1] != 0 or bbList[i,-1] == -1:
                i+=1
            else:
                # create an empty arr for storing indexes when a neighboring bb is found
                idx_arr = []

                for j,bb2 in enumerate(bbList):
                    flag = False
                    # do not iterate same item again
                    if j==i:
                        pass
                    else:
                        # check if the rules are met, if met increment count by one and append the index of 
                        # box to index array.
                        flag = checkRules(bb1,bb2)
                        if checkRules(bb1,bb2):
                            idx_arr.append(j)
                            count += 1   
                # after iterating through all items but the current one, 
                # see how many counts there is, meaning that how many neighboring digits 
                # there are for the current bounding box
                if count == 1:
                    # If 1 mark it as -2
                    bbList[i,-1]=-2
                if count == 2:
                    # If there are 2 digits meaning that current digit is the middle one of 3
                    # create an array
                    temp = np.zeros(shape=(3,5))
                    bbList[i,-1] = c
                    # initialize first 4 items as x,y,w,h of 3 digits found.
                    temp[0,:4] = bbList[i,:4]
                    temp[0,4] = bbList[i,6]
                    # iterate index_arr to get information of corresponding bounding boxes in bbList.
                    for ii,idx in enumerate(idx_arr):
                        temp[ii+1,:4] = bbList[idx,:4]
                        bbList[idx,-1] = c
                        temp[ii+1,4] = bbList[idx,6]
                    # find min - max values of x-y to draw new bounding box
                    minY = int(np.min(temp[:,1]))-2
                    maxY = int(temp[np.argmax(temp[:,1]),3]+np.max(temp[:,1]))
                    maxX = int(temp[np.argmax(temp[:,1]),2]+np.max(temp[:,0]))
                    minX = int(np.min(temp[:,0]))
                    lbl = ''
                    if arrow_cnt is not None:
                        # find arrows
                        for n,arrow in enumerate(arrow_bbs):
                            x,y,w,h = arrow
                            cY = y+h*0.5
                            # add arrow to the bounding box if it meets the condition
                            if minY<=cY and cY<=maxY and np.abs(maxX-x)<1.5*w and maxX<x:
                                arrow_counter += 1
                                lbl = arrow_labels[n]
                                if int(arrow_labels[n]) == 1:
                                    lbl = ' (L)'
                                else:
                                    lbl = ' (R)'
                                maxX = int(x+w)+2
                                # rearrange coordinates
                                if y+h+5>maxY:
                                    maxY = int(y+h)+5
                                if y-5<minY:
                                    minY = int(y)-5
                    # sort numbers inside the new bounding box by X values
                    # since we read from left to right
                    x_vals = np.argsort(temp[:,0])
                    if write:
                        imgg = img.copy()
                        if not (os.path.exists('output') or os.path.exists(str(task_num))):
                            os.makedirs('output')
                            os.makedirs(os.path.join('output',str(task_num)))
                        cropped = imgg[minY:maxY,minX:maxX]
                        rand_num = random.randint(0,9999)
                        rand_2 = random.randint(0,9999)
                        filename = os.path.join(os.path.join(os.path.join(BASE_PATH,'output'),str(task_num)),'output_%d_%d.jpg'%(rand_num,rand_2))
                        cv2.imwrite(filename,cropped)
                    label = ""
                    for idx in x_vals:
                        label += str(temp[idx,4])[0]
                    # visualize
                    label += lbl
                    cv2.rectangle(temp_img,(minX,minY),(maxX+5,maxY+5),[0,255,0],2)
                    resulted_boxes.append((minX,minY,maxX,maxY))
                    cv2.putText(temp_img,label,(int((minX+maxX)*0.5),minY-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,[0,0,255])
                    c+=1
                    labels.append(label)
                if count==0:
                    bbList[i,-1] = -1
                i+=1    
        return bbList,temp_img,resulted_boxes,labels,arrow_counter
    except:
        print('Can not draw bounding boxes because of an error, instead drawing individual bounding boxes')
        _,_,resulted_img = getMostContour(img,digit_svm,digit_knn,[Gaussian_Morph(img,9,3,250,300)],True,wThresh=9,hThresh=21)
        return -1,resulted_img,resulted_boxes   


def checkRules(bb1,bb2):
    minBoxHeight = np.min(np.array([bb1[3],bb2[3]]))
    if(np.abs(bb1[3]-bb2[3]) < minBoxHeight  and np.abs(bb1[5]-bb2[5])<0.7*minBoxHeight and np.abs(bb1[4]-bb2[4])<=1.3*minBoxHeight):
        return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-show', "--showFilters", help='show each filter applied on image',type=bool)
    parser.add_argument('-i', "--imgFolder", help='Image folder name')
    parser.add_argument('-r','--resultsOnly',help='Show results only',type=bool)

    try:
        args = parser.parse_args()
    except argparse.ArgumentError() as e:
        print(e)

    cell_size = (5, 5)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins

    # this approach is based on pixelou's answer in this link : https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python

    # winSize is the size of the image cropped to a multiple of the cell size
    # HOG ( Histogram of Oriented Gradients) method will be used instead of 
    # raw pixel values for training, since it provides more useful information
    # about images. Also increases accuracy a great amount.
    hog = cv2.HOGDescriptor(_winSize=(25 // cell_size[1] * cell_size[1],
                                        25 // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)

    n_cells = (25 // cell_size[0], 25 // cell_size[1])
    # finally, read images and show results.
    # ML models initalization for digit recognition
     # train the algorithm for digits
    digit_svm = cv2.ml.SVM_create()
    digit_knn = cv2.ml.KNearest_create()
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
    if args.imgFolder is not None:
        imgpath = os.path.join(BASE_PATH,args.imgFolder)
    imgpth = os.path.join(os.path.join(os.path.join('home','student'),'test'),'task1')
    imgpath = os.path.join(BASE_PATH,imgpth)
    bs_text = []
    for imgpth in glob.glob(os.path.join(imgpath,'*.jpg')):
        base = os.path.basename(imgpth)
        file_name = os.path.splitext(base)[0]
        img = cv2.imread(imgpth)
        img = cv2.resize(img, (540, 720))  
        gaussian_filter = Gaussian_Morph(img,9,3,250,300)
        white_filter = filterWithWhite(img,70)
        filterArr = [gaussian_filter,white_filter]
        if args.showFilters and not args.resultsOnly:
            for flt in filterArr:
                showWait(flt,'filters used for digits')
        dig_contours,dig_labels,dig_img = getMostContour(img,digit_svm,digit_knn,filterArr,digits=True,wThresh=9,hThresh=21) # 9 -21
        bbList,result,resulted_boxes,bs_labels,_ = mergeBoundingBoxes(img.copy(),dig_contours,dig_labels,write=True)
        # append resulted labels to a list
        if len(bs_labels)>0:
            bs_text.append(file_name+': '+str(bs_labels[0])+'\n')
        if args.resultsOnly:
            showWait(result,'result')
    pth = os.path.join(os.path.join(os.path.abspath(''),'output'),str(task_num))
    if os.path.exists(os.path.join(pth,"BuildingAnswer.txt")):
        os.remove("BuildingAnswer.txt")
    bs_file = open('BuildingAnswer.txt','w')
    bs_file.writelines(bs_text)
    bs_file.close()