import sys
import os
import cv2
import numpy as np
import JPEGEncoder as en

def stitchPair(img1, img2, opt):
    detector = cv2.ORB_create()
    # detector = cv2.AKAZE_create()
    imgGray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    ret1, mask1 = cv2.threshold(imgGray1, 1, 255, cv2.THRESH_BINARY)
    # find key points
    kp1, des1 = detector.detectAndCompute(imgGray1,None)

    imgGray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret2, mask2 = cv2.threshold(imgGray2, 1, 255, cv2.THRESH_BINARY)
    # find key points
    kp2, des2 = detector.detectAndCompute(imgGray2,None)

    # Visualize matching procedure
    # print("keypoints: {}, descriptor: {}".format(len(kp1), des1.shape))
    keypoints1Im = cv2.drawKeypoints(img1, kp1, None)
    # cv2.imwrite("KEYPOINTS1.png",keypoints1Im)
    # print("keypoints: {}, descriptor: {}".format(len(kp2), des2.shape))
    keypoints2Im = cv2.drawKeypoints(img2, kp2, None)
    # cv2.imwrite("KEYPOINTS2.png",keypoints2Im)

    match = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = match.knnMatch(des1,des2,k=2)

    #prune bad matches
    good = []
    for m,n in matches:
        # if m.distance < 0.55 * n.distance:
        if m.distance < float(opt) * n.distance:
            good.append(m)

    # print (str(len(good)) + " Matches were Found")

    if len(good) <= 10:
        return img1
            
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags = 2)
    # cv2.imwrite("matches.png", img3)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    A = cv2.estimateRigidTransform(src_pts,dst_pts,fullAffine=False)

    if A is None:
      HomogResult = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC)
      H = HomogResult[0]

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    corners1 = np.float32(([0,0],[0,h1],[w1, h1], [w1,0]))
    corners2 = np.float32(([0,0],[0,h2],[w2, h2], [w2,0]))
    
    warpedCorners2= np.zeros((4,2))

    for i in range(0,4):
        cornerX = corners2[i, 0]
        cornerY = corners2[i, 1]
        if A is not None:
            warpedCorners2[i, 0] = A[0,0]*cornerX + A[0,1]*cornerY + A[0,2]
            warpedCorners2[i, 1] = A[1,0]*cornerX + A[1,1]*cornerY + A[1,2]
        else:
            warpedCorners2[i, 0] = (H[0,0]*cornerX + H[0,1]*cornerY + H[0,2]) / (H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
            warpedCorners2[i, 1] = (H[1,0]*cornerX + H[1,1]*cornerY + H[1,2]) / (H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])

    allCorners = np.concatenate((corners1, warpedCorners2), axis=0)

    [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

    translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))

    warpedResImg = cv2.warpPerspective(img1, translation, (xMax-xMin, yMax-yMin))
    
    if (A is None):
        fullTransformation = np.dot(translation, H)
        warpedImage2 = cv2.warpPerspective(img2, fullTransformation, (xMax-xMin, yMax-yMin))
    else:
        warpedImageTemp = cv2.warpPerspective(img2, translation, (xMax-xMin, yMax-yMin))
        warpedImage2 = cv2.warpAffine(warpedImageTemp, A, (xMax-xMin, yMax-yMin))

    result = np.where(warpedImage2 != 0, warpedImage2, warpedResImg)
    
    # resGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # warpedResGray = cv2.warpPerspective(resGray, translation, (xMax - xMin, yMax - yMin))
    # ret, mask1 = cv2.threshold(warpedResGray, 1, 255, cv2.THRESH_BINARY_INV)
    # mask3 = np.float32(mask1)/255

    # warpedImage2[:,:,0] = warpedImage2[:,:,0] * mask3
    # warpedImage2[:,:,1] = warpedImage2[:,:,1] * mask3
    # warpedImage2[:,:,2] = warpedImage2[:,:,2] * mask3

    # result = warpedResImg + warpedImage2
    
    return result
    pass

def Combinar(image_dir, key_opt):
    dir_list = os.listdir(image_dir)
    try:
        dir_list.remove('.DS_Store')
    except: ValueError

    dir_list = list(map(lambda x: os.path.join(image_dir, x), dir_list))
    print(dir_list)
    dir_list.sort()
    print(dir_list)
    for image in dir_list[:1]:
        # result = cv2.imread(image)
        result = en.compress(cv2.imread(image))
    i = 0
    for image in dir_list[1:]:
        i+=1
        print("Processing - ",image)
        result = stitchPair(result, en.compress(cv2.imread(image)), key_opt)
        cv2.imwrite("output_dir/int_res" + str(i) + ".png", result)
        # result = stitchPair(result, cv2.imread(image), key_opt)
        # h, w = result.shape[:2]
        # if h > 4000 and w > 4000:
        #     if h > 4000:
        #         h = h * (4000/h)
        #         w = w * (4000/h)
        #     elif w > 4000:
        #         h = h * (4000/w)
        #         w = w * (4000/w)
        #     result = cv2.resize(result, (int(w),int(h)))
        #     cv2.imwrite("result.png", result)
        #     pass
        cv2.imwrite("result.png", result)
        pass
    pass

if __name__ == "__main__":
    if (len(sys.argv)<3):
        print >> sys.stderr, ("Usage: %s <image_dir> <opt>" % sys.argv[0])
        sys.exit(-1)
        pass
    # img1 = cv2.imread('DJI_0874.JPG')
    # img2 = cv2.imread('DJI_0875.JPG')
    # result = stitchPair(img1, img2)
    # cv2.imwrite("Testresult.png", result)
    # Stitch mas de dos
    Combinar(sys.argv[1], sys.argv[2])
    print (">>> Done <<<")
    pass