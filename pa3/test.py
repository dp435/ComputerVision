import unittest
import numpy as np
import sys, os, imp
import cv2
import scipy
import blend

class Blend(unittest.TestCase):
	# identity matrix: no change
    def testImageBoundingBox1(self):
        img = np.zeros((101,51,4))
        M = np.eye(3)

        actual_result = blend.imageBoundingBox(img, M)
        expected_result = (0,0,50,100)
        self.assertEqual(actual_result, expected_result)

	# translation matrix: shift x by 10
    def testImageBoundingBox2(self):
        img = np.zeros((101,51,3))
        M = np.eye(3)
        M[0,2] = 10

        actual_result = blend.imageBoundingBox(img, M)
        expected_result = (10,0,60,100)
        self.assertEqual(actual_result, expected_result)

	# translation matrix: shift x by 10, shift y by 50
    def testImageBoundingBox3(self):
        img = np.zeros((101,51,3))
        M = np.eye(3)
        M[0,2] = 10
        M[1,2] = 50

        actual_result = blend.imageBoundingBox(img, M)
        expected_result = (10,50,60,150)
        self.assertEqual(actual_result, expected_result)

	# scale matrix: scale y by 2
    def testImageBoundingBox4(self):
        img = np.zeros((101,51,3))
        M = np.eye(3)
        M[1,1] = 2

        actual_result = blend.imageBoundingBox(img, M)
        expected_result = (0,0,50,200)
        self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
    unittest.main()