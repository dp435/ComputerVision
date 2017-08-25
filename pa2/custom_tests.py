import unittest
import numpy as np
import sys, os, imp
import cv2
import scipy
import transformations
import features
import traceback
from PIL import Image
import features

class SSDMatching(unittest.TestCase):
	# sanity check: no feature matches.
	def test_empty(self):
		expected_result = []
		self.assertEqual(SSDFM.matchFeatures(np.zeros((0,2)),np.zeros((0,2))), expected_result)
	
	# Test for 1 element in each feature list.
	def test_oneElement(self):
		desc1 = np.array([[1,2,3]])
		desc2 = np.array([[1,2,3]])
		expected_result = cv2.DMatch(0, 0, 0.)
		actual_result = SSDFM.matchFeatures(desc1,desc2)[0]
		self.assertEqual(actual_result.queryIdx, expected_result.queryIdx)
		self.assertEqual(actual_result.trainIdx, expected_result.trainIdx)
		self.assertEqual(actual_result.distance, expected_result.distance)

	# Test for 2 elements where ties are not present.
	def test_multiMatchNoTie(self):
		desc1 = np.array([[1,2,3],[0,0,0]])
		desc2 = np.array([[1,1,1],[1,2,3]])
		expected_result = cv2.DMatch(0, 1, 0.)
		actual_result = SSDFM.matchFeatures(desc1,desc2)[0]
		self.assertEqual(actual_result.queryIdx, expected_result.queryIdx)
		self.assertEqual(actual_result.trainIdx, expected_result.trainIdx)
		self.assertEqual(actual_result.distance, expected_result.distance)

		expected_result = cv2.DMatch(1, 0, np.min(scipy.spatial.distance.cdist(desc1, desc2)[1]))
		actual_result = SSDFM.matchFeatures(desc1,desc2)[1]
		self.assertEqual(actual_result.queryIdx, expected_result.queryIdx)
		self.assertEqual(actual_result.trainIdx, expected_result.trainIdx)
		self.assertEqual(actual_result.distance, expected_result.distance)

	# Test for 3 elements where ties are present: only distances are deterministic.
	def test_multiMatchTie(self):
		desc1 = np.array([[1,2,3],[1,0,0],[1,0,3]])
		desc2 = np.array([[1,2,0],[1,2,3],[1,2,3]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)
		expected_result = ([cv2.DMatch(0, 1, 0.),
			cv2.DMatch(1, 0, np.min(SSD[1])),
			cv2.DMatch(2, 1, np.min(SSD[2]))])
		actual_result = SSDFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)
		self.assertEqual(actual_result[2].distance, expected_result[2].distance)

	# Test when # of features detected in Image 1 > Image 2.
	def test_unevenMatch1(self):
		desc1 = np.array([[1,2,3],[1,0,0],[1,2,10]])
		desc2 = np.array([[1,2,0],[1,0,0]])

		SSD = scipy.spatial.distance.cdist(desc1, desc2)
		expected_result = ([cv2.DMatch(0, 0, np.min(SSD[0])),
			cv2.DMatch(1, 1, np.min(SSD[1])),
			cv2.DMatch(2, 0, np.min(SSD[2]))])
		actual_result = SSDFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)
		self.assertEqual(actual_result[2].queryIdx, expected_result[2].queryIdx)
		self.assertEqual(actual_result[2].trainIdx, expected_result[2].trainIdx)
		self.assertEqual(actual_result[2].distance, expected_result[2].distance)

	# Test when # of features detected in Image 1 < Image 2.
	def test_unevenMatch2(self):
		desc1 = np.array([[1,2,3],[1,0,1]])
		desc2 = np.array([[1,2,0],[1,2,1],[1,2,3]])

		SSD = scipy.spatial.distance.cdist(desc1, desc2)
		expected_result = ([cv2.DMatch(0, 2, np.min(SSD[0])),
			cv2.DMatch(1, 1, np.min(SSD[1]))])
		actual_result = SSDFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)


class RatioMatching(unittest.TestCase):
	# sanity check: no feature matches.
	def test_empty(self):
		expected_result = []
		self.assertEqual(RFM.matchFeatures(np.zeros((0,2)),np.zeros((0,2))), expected_result)

	# Test for 1 element in each feature list.
	def test_oneElement(self):
		desc1 = np.array([[1,2,3]])
		desc2 = np.array([[1,2,3]])
		expected_result = cv2.DMatch(0, 0, 0.)
		actual_result = RFM.matchFeatures(desc1,desc2)[0]
		self.assertEqual(actual_result.queryIdx, expected_result.queryIdx)
		self.assertEqual(actual_result.trainIdx, expected_result.trainIdx)
		self.assertEqual(actual_result.distance, expected_result.distance)

	# Test when # of features detected in Image 1 > (# of features detected in Image 2 == 1).
	def test_unevenMatch1(self):
		desc1 = np.array([[1,2,3],[1,0,0],[1,2,10]])
		desc2 = np.array([[1,2,0]])

		expected_result = ([cv2.DMatch(0, 0, 0.),
			cv2.DMatch(1, 0, 0.),
			cv2.DMatch(2, 0, 0.)])
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)
		self.assertEqual(actual_result[2].queryIdx, expected_result[2].queryIdx)
		self.assertEqual(actual_result[2].trainIdx, expected_result[2].trainIdx)
		self.assertEqual(actual_result[2].distance, expected_result[2].distance)

	# PERFECT MATCHING: Test when (# of features detected in Image 1 == 1) < (# of features detected in Image 2).
	def test_unevenPerfectMatch(self):
		desc1 = np.array([[1,2,3]])
		desc2 = np.array([[1,2,3],[1,0,0],[1,20,30]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)

		expected_result = [cv2.DMatch(0, 0, 0.)]
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)

	# IMPERFECT MATCHING: Test when (# of features detected in Image 1 == 1) < (# of features detected in Image 2).
	def test_unevenImperfectMatch1(self):
		desc1 = np.array([[1,2,3]])
		desc2 = np.array([[1,2,4],[1,0,0],[1,20,30]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)

		expected_result = [cv2.DMatch(0, 0, np.divide(SSD[0,0],SSD[0,1]))]
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)

	# IMPERFECT MATCHING: Test when (# of features detected in Image 1 == 1) < (# of features detected in Image 2).
	def test_unevenImperfectMatch2(self):
		desc1 = np.array([[11,21,31],[1,2,3]])
		desc2 = np.array([[1,2,4],[1,0,0],[1,20,30]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)

		expected_result = ([cv2.DMatch(0, 2, np.divide(SSD[0,2],SSD[0,0])),
			cv2.DMatch(1, 0, np.divide(SSD[1,0],SSD[1,1]))])
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(len(actual_result), len(expected_result))
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)

	# Test for 3 elements where ties are not present.
	def test_multiMatchNoTie(self):
		desc1 = np.array([[1,2,3,4],[0,0,0,0],[4,3,2,1]])
		desc2 = np.array([[1,1,1,1],[1,2,3,0],[5,4,3,2]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)

		expected_result = ([cv2.DMatch(0, 0, np.divide(SSD[0,0],SSD[0,1])),
			cv2.DMatch(1, 0, np.divide(SSD[1,0],SSD[1,1])),
			cv2.DMatch(2, 2, np.divide(SSD[2,2],SSD[2,1]))])
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)
		self.assertEqual(actual_result[2].queryIdx, expected_result[2].queryIdx)
		self.assertEqual(actual_result[2].trainIdx, expected_result[2].trainIdx)
		self.assertEqual(actual_result[2].distance, expected_result[2].distance)

	# Test for 3 elements where ties are present.
	def test_multiMatchTie(self):
		desc1 = np.array([[1,2,3,2],[0,0,0,0],[4,3,2,1]])
		desc2 = np.array([[1,2,3,1],[1,2,3,3],[5,4,3,2]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)

		expected_result = ([cv2.DMatch(0, 0, np.divide(SSD[0,0],SSD[0,1])),
			cv2.DMatch(1, 0, np.divide(SSD[1,0],SSD[1,1])),
			cv2.DMatch(2, 2, np.divide(SSD[2,2],SSD[2,0]))])
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)
		self.assertEqual(actual_result[2].queryIdx, expected_result[2].queryIdx)
		self.assertEqual(actual_result[2].trainIdx, expected_result[2].trainIdx)
		self.assertEqual(actual_result[2].distance, expected_result[2].distance)

	# Test for when multiple elements match to the same feature.
	def test_multiMatch(self):
		desc1 = np.array([[1,2,3,4],[1,2,3,0],[1,1,2,2]])
		desc2 = np.array([[1,1,1,1],[1,2,3,0],[5,4,3,2]])
		SSD = scipy.spatial.distance.cdist(desc1, desc2)

		expected_result = ([cv2.DMatch(0, 0, np.divide(SSD[0,0],SSD[0,1])),
			cv2.DMatch(1, 1, np.divide(SSD[1,1],SSD[1,0])),
			cv2.DMatch(2, 0, np.divide(SSD[2,0],SSD[2,1]))])
		actual_result = RFM.matchFeatures(desc1,desc2)
		self.assertEqual(actual_result[0].queryIdx, expected_result[0].queryIdx)
		self.assertEqual(actual_result[0].trainIdx, expected_result[0].trainIdx)
		self.assertEqual(actual_result[0].distance, expected_result[0].distance)
		self.assertEqual(actual_result[1].queryIdx, expected_result[1].queryIdx)
		self.assertEqual(actual_result[1].trainIdx, expected_result[1].trainIdx)
		self.assertEqual(actual_result[1].distance, expected_result[1].distance)
		self.assertEqual(actual_result[2].queryIdx, expected_result[2].queryIdx)
		self.assertEqual(actual_result[2].trainIdx, expected_result[2].trainIdx)
		self.assertEqual(actual_result[2].distance, expected_result[2].distance)


if __name__ == '__main__':
	SSDFM = features.SSDFeatureMatcher()
	RFM = features.RatioFeatureMatcher()

	unittest.main()