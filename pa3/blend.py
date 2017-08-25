import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN

    (height, width, channel) = img.shape

    corner = np.zeros((4,3))

    # bottom-left corner
    corner[0,0] = 0.0           #x
    corner[0,1] = 0.0           #y
    corner[0,2] = 1.0           #z

    # bottom-right corner
    corner[1,0] = width-1.0     #x
    corner[1,1] = 0.0           #y
    corner[1,2] = 1.0           #z

    # top-right corner
    corner[2,0] = width-1.0     #x
    corner[2,1] = height-1.0    #y
    corner[2,2] = 1.0           #z

    # top-left corner
    corner[3,0] = 0.0           #x
    corner[3,1] = height-1.0    #y
    corner[3,2] = 1.0           #z

    for k in range(4):
        corner[k] = M.dot(corner[k].reshape(3,1)).flatten()
        corner[k] = corner[k]/corner[k,2]

    x_column = corner[:,0]
    y_column = corner[:,1]

    minX = np.amin(x_column)
    maxX = np.amax(x_column)
    minY = np.amin(y_column)
    maxY = np.amax(y_column)

    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    # dimensions of images
    (img_height, img_width, img_channel) = img.shape
    (acc_height, acc_width, acc_channel) = acc.shape

    # inverse transform matrix
    inv_M = np.linalg.inv(M)

    # bounding box of img
    (minX, minY, maxX, maxY) = imageBoundingBox(img, M)

    for h  in range(minY,maxY):
        for w in range(minX,maxX):

            pixel = np.array([w,h,1]).reshape(3,1)
            src_pixel = inv_M.dot(pixel)

            # normalize to cartesian coordinates
            x = float(src_pixel[0])/float(src_pixel[2])
            y = float(src_pixel[1])/float(src_pixel[2])

            # check for out-of-bounds
            if (x < 0 or x > img_width-1 or y < 0 or y > img_height-1):
                continue

            x_floor = int(np.floor(x))
            x_ceil = int(np.ceil(x))
            y_floor = int(np.floor(y))
            y_ceil = int(np.ceil(y))

            # check for black pixels
            if ((img[y_floor, x_floor, 0] == 0 and img[y_floor, x_floor, 1] == 0 and img[y_floor, x_floor, 2] == 0)
                or (img[y_floor, x_ceil, 0] == 0 and img[y_floor, x_ceil, 1] == 0 and img[y_floor, x_ceil, 2] == 0)
                or (img[y_ceil, x_ceil, 0] == 0 and img[y_ceil, x_ceil, 1] == 0 and img[y_ceil, x_ceil, 2] == 0)
                or (img[y_ceil, x_floor, 0] == 0 and img[y_ceil, x_floor, 1] == 0 and img[y_ceil, x_floor, 2] == 0)):
                continue

            # check to see if blendWidth > image width
            if (blendWidth > img_width/2.0):
                blendWidth = img_width/2.0

            # check to see if current pixel is within a distance of 'blendWidth' from either of the borders
            loc = min(w-minX, maxX-w)
            alpha = 1.0
            if (loc < blendWidth):
                alpha = float(loc) / float(blendWidth)

            # lands on non-floating point coordinate
            if (x % 1.0 == 0.0 and y % 1.0 == 0.0):
                for c in range(3):
                    acc[h,w,c] += alpha*img[int(y),int(x),c]
            
            # only x is non-floating point.
            elif (x % 1.0 == 0.0 and y % 1.0 != 0.0):
                c1 = (1.0-(y-float(y_floor)))
                c2 = (1.0-(float(y_ceil)-y))
                for c in range(3):
                    acc[h,w,c] += alpha*(c1*img[y_floor,int(x),c] + c2*img[y_ceil,int(x),c])

            # only y is non-floating point.
            elif (x % 1.0 != 0.0 and y % 1.0 == 0.0):
                c1 = (1.0-(x-float(x_floor)))
                c2 = (1.0-(float(x_ceil)-x))
                for c in range(3):
                    acc[h,w,c] += alpha*(c1*img[int(y),x_floor,c] + c2*img[int(y),x_ceil,c])

            # bilinear interpolation
            else:
                c1 = (1.0-(x-float(x_floor))) * (1.0-(y-float(y_floor)))
                c2 = (1.0-(float(x_ceil)-x)) * (1.0-(y-float(y_floor)))
                c3 = (1.0-(float(x_ceil)-x)) * (1.0-(float(y_ceil)-y))
                c4 = (1.0-(x-float(x_floor))) * (1.0-(float(y_ceil)-y))
                for c in range(3):
                    acc[h,w,c] += alpha*((c1*img[y_floor,x_floor,c] + c2*img[y_floor,x_ceil,c]
                        + c3*img[y_ceil,x_ceil,c] + c4*img[y_ceil,x_floor,c]))

            # accumulate alpha in alpha channel
            acc[h,w,3] += alpha

    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN

    (height, width, channel) = acc.shape 
    img = np.zeros((height,width,channel), dtype='uint8')

    for h in range(height):
        for w in range(width):
            alpha = float(acc[h,w,3])

            if (alpha > 0):
                img[h,w] = acc[h,w]/acc[h,w,3]
            else:
                img[h,w] = 0

            img[h,w,3] = 255

    #TODO-BLOCK-END
    # END TODO
    return img


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN

        (minX_prime, minY_prime, maxX_prime, maxY_prime) = imageBoundingBox(img, M)
        minX = min(minX, minX_prime)
        minY = min(minY, minY_prime)
        maxX = max(maxX, maxX_prime)
        maxY = max(maxY, maxY_prime)

        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

        # First image
        if count == 0:
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            p = np.array([0.5 * width, 0, 1])
            p = M_trans.dot(p)
            x_final, y_final = p[:2] / p[2]

    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does inverse mapping which means A is an affine
    # transform that maps final panorama coordinates to accumulator coordinates
    #TODO-BLOCK-BEGIN

    if is360:
        A[0,2] = -float(width)/2.0
        A[1,0] = -(float(y_init)-float(y_final))/(float(x_init)-float(x_final))

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

