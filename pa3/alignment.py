import math
import cv2
import numpy as np
import random

eTranslate = 0
eHomography = 1

def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature 
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9 
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        A[2*i, 0] = a_x
        A[2*i, 1] = a_y
        A[2*i, 2] = 1
        A[2*i, 6] = -b_x*a_x
        A[2*i, 7] = -b_x*a_y
        A[2*i, 8] = -b_x

        A[2*i+1, 3] = a_x
        A[2*i+1, 4] = a_y
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = -b_y*a_x
        A[2*i+1, 7] = -b_y*a_y
        A[2*i+1, 8] = -b_y

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    num_eigs = s.shape[0]

    H[0][0] = Vt[8,0]/Vt[8,8];
    H[0][1] = Vt[8,1]/Vt[8,8];
    H[0][2] = Vt[8,2]/Vt[8,8];
    H[1][0] = Vt[8,3]/Vt[8,8];
    H[1][1] = Vt[8,4]/Vt[8,8];
    H[1][2] = Vt[8,5]/Vt[8,8];
    H[2][0] = Vt[8,6]/Vt[8,8];
    H[2][1] = Vt[8,7]/Vt[8,8];
    H[2][2] = Vt[8,8]/Vt[8,8];

    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''    

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of 
    #motion models, pure translations (m == eTranslation) and 
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as 
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN

    M = np.eye(3)
    bestCount = 0
    sampleQuantity = 0

    # Determine which motion model to handle.
    if (m == eTranslate):
        sampleQuantity = 1
    elif (m == eHomography):
        sampleQuantity = 4

    # Perform RANSAC

    for k in range(nRANSAC):
        sampledPoints = random.sample(range(0,len(matches)),sampleQuantity)

        # Generate inter-image transformation matrix using random sample.
        interM = leastSquaresFit(f1, f2, matches, m, sampledPoints)

        inlier_indices = getInliers(f1, f2, matches, interM, RANSACthresh)

        if (len(inlier_indices) > bestCount):
            M = leastSquaresFit(f1, f2, matches, m, inlier_indices)
            bestCount = len(inlier_indices)

    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed 
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN

        (f1_x, f1_y) = f1[matches[i].queryIdx].pt
        (f2_x, f2_y) = f2[matches[i].trainIdx].pt

        f1_transformed = np.array([f1_x, f1_y, 1])
        f1_transformed = M.dot(f1_transformed)

        f1_x = np.divide(float(f1_transformed[0]),float(f1_transformed[2]))
        f1_y = np.divide(float(f1_transformed[1]),float(f1_transformed[2]))

        if (((f1_x-f2_x)**2.0 + (f1_y-f2_y)**2.0) < RANSACthresh**2.0):
            inlier_indices.append(i)

        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate) 
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a 
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN

            (x_i, y_i) = f1[matches[inlier_indices[i]].queryIdx].pt
            (xprime_i, yprime_i) = f2[matches[inlier_indices[i]].trainIdx].pt
            u += (xprime_i - x_i)
            v += (yprime_i - y_i)

            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN

        matchesOfInterest = []
        for i in range(len(inlier_indices)):
            matchesOfInterest.append(matches[inlier_indices[i]])

        M = computeHomography(f1, f2, matchesOfInterest)

        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

