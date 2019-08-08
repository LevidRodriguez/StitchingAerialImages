import cv2
import numpy as np
import copy

img1 = cv2.imread('0.bmp')
imgGray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('2.bmp')
imgGray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create()
# detector = cv2.AKAZE_create()
# find key points
kp1, des1 = detector.detectAndCompute(imgGray1,None)
kp2, des2 = detector.detectAndCompute(imgGray2,None)

# Visualize matching procedure
print("keypoints: {}, descriptor: {}".format(len(kp1), des1.shape))
keypoints1Im = cv2.drawKeypoints(img1, kp1, None)
cv2.imwrite("KEYPOINTS1.png",keypoints1Im)
print("keypoints: {}, descriptor: {}".format(len(kp2), des2.shape))
keypoints2Im = cv2.drawKeypoints(img2, kp2, None)
cv2.imwrite("KEYPOINTS2.png",keypoints2Im)

match = cv2.BFMatcher_create(cv2.NORM_HAMMING)
matches = match.knnMatch(des1,des2,k=2)

#prune bad matches
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)
matches = copy.copy(good)
        
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags = 2)

cv2.imwrite("matches.png", img3)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

A = cv2.estimateRigidTransform(src_pts,dst_pts,fullAffine=False)

if (A == None):
  HomogResult = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC)
  H = HomogResult[0]

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]

corners1 = np.float32(([0,0],[0,h1],[w1, h1], [w1,0]))
corners2 = np.float32(([0,0],[0,h2],[w2, h2], [w2,0]))
warpedCorners2= np.zeros((4,2))

