import os
import sys
import cv2
import math
import numpy as np
from numpy import linalg

class Stitch(object):
    def __init__(self, image_dir, key_frame, output_dir, image_filter=None):
        '''  ***+++*** '''
        # self.key_frame_
        print("Hola Mundo!")
        pass
    def stitch(self, base_image_rgb, round=0):
        pass
    pass

if __name__ == '__main__':
    if ( len(sys.argv) < 4 ):
        print ("Usage: %s <image_dir> <key_frame> <output>" % sys.argv[0])
        sys.exit(-1)
    Stitch(sys.argv[1], sys.argv[2], sys.argv[3])