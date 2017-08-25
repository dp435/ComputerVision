import unittest
import numpy as np
import sys, os, imp
import cv2
import scipy
import alignment
import blend

class Alignment(unittest.TestCase):
    def testComputeHomography1(self):

        f1 = []
        f2 = []

        f1.append(cv2.KeyPoint(x=2,y=2,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))
        f1.append(cv2.KeyPoint(x=5,y=2,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))
        f1.append(cv2.KeyPoint(x=4,y=4,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))
        f1.append(cv2.KeyPoint(x=2,y=5,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))

        f2.append(cv2.KeyPoint(x=2,y=2,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))
        f2.append(cv2.KeyPoint(x=5,y=2,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))
        f2.append(cv2.KeyPoint(x=4,y=4,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))
        f2.append(cv2.KeyPoint(x=2,y=5,_size=2,_angle=0,_response=0,_octave=0,_class_id=0))

        matches = []
        for i in range(4):
            matches.append(cv2.DMatch(i,i,0.0))
        print alignment.computeHomography(f1, f2, matches)

if __name__ == '__main__':
    unittest.main()