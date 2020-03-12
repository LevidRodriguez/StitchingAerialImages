import Combiner
import cv2
import _Dataset as Dataset
import os
import datetime
import shutil

# 06/03/2020
import sys

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print >> sys.stderr, ("Usage: %s [image_dir_input] [image_dir_output]" % sys.argv[0])
        sys.exit(-1)
    _Dataset = Dataset.Dataset(sys.argv[1])
    _Dataset.write(sys.argv[1])
    if os.path.isdir('results') == True:
        shutil.rmtree('results', ignore_errors=True, onerror=None)
    
    os.mkdir('results')
    fileName = "datasets/imageData.txt"
    imageDirectory = "datasets/images/"
    
    if os.path.isdir('temp') == True:
        shutil.rmtree('temp', ignore_errors=False, onerror=None)
    os.mkdir('temp')

    print("Copying Images to Temp Directory")
    allImages, dataMatrix = _Dataset.importData(fileName, imageDirectory)
    _Dataset.changePerpective(allImages, dataMatrix)
    results = Combiner.combine()
    cv2.imwrite("results/finalResult.png",results)

    print("Good!")
    pass