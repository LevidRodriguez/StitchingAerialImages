import ExifData, XMPData
import glob
from PIL import Image
import os
import numpy as np
import cv2
import geometry as gm
import gc

class Dataset:
    # Contructor
    def __init__(self, input_images_dir):
        # Atributos
        self.input = open("datasets/imageData.txt","w+")
        self.input.close()
        self.input = open("datasets/imageData.txt","a+")
        pass
    
    # Metodos
    def write(self, images_dir):
        for image in sorted(glob.glob(images_dir+'*')):
            exif_data = ExifData.get_exif_data(Image.open(image))
            lat, lon = ExifData.get_lat_lon(exif_data)
            alt, roll, yaw, pitch = XMPData.xmp(image)
            #print lon, lat, alt, yaw, pitch, roll
            st = (os.path.basename(image)) + "," + str(float(lon)) + "," + str(float(lat)) + "," + str(float(alt)) + "," + str(float(yaw)) + "," + str(float(pitch)) + "," + str(float(roll)) + "\n"
            self.input.write(st)
        self.input.close()
        pass
    
    def importData(self, fileName, imageDirectory):
        allImages = []
        dataMatrix = np.genfromtxt(fileName,delimiter=",",usecols=range(1,7),dtype=float) #read numerical data
        fileNameMatrix = np.genfromtxt(fileName,delimiter=",",usecols=[0],dtype=str) #read filen name strings
        for i in range(0, fileNameMatrix.shape[0]):
            cv2.imwrite("temp/" + str(i).zfill(4) + ".png", cv2.imread(imageDirectory + fileNameMatrix[i]))
        return allImages, dataMatrix
        pass
    
    def changePerpective(self, imageList, dataMatrix):
        images = sorted(glob.glob("temp/*.png"))
        for i  in range(0, len(images)):
            image = cv2.imread(images[i])
            image = image[::2, ::2, :]
            M = gm.computeUnRotMatrix(dataMatrix[i, :])
            correctedImage = gm.warpPerspectiveWithPadding(image,M)
            cv2.imwrite("temp/" + str(i).zfill(4) + ".png", correctedImage)
        pass
    pass