import cv2
import numpy as np

def zero_pad(matrix, pad_height, pad_width):
    '''Pads the top and bottom-side of inputted matrix with <pad_height> number of zeros,
    and the left and right-side with <pad_width> number of zeros.'''
    if (matrix.ndim == 2):
        (height, width) = matrix.shape
        padded_matrix = np.zeros(shape = (height+2*pad_height, width+2*pad_width))
        padded_matrix[pad_height:pad_height+height,pad_width:pad_width+width] = matrix
        return padded_matrix
    else:
        (height, width, dim) = matrix.shape
        padded_matrix = np.zeros(shape = (height+2*pad_height, width+2*pad_width, dim))
        for d in range(dim):
            padded_matrix[pad_height:pad_height+height,pad_width:pad_width+width,d] = matrix[:,:,d]
        return padded_matrix


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    if (img.ndim == 2):
        G = np.zeros(shape = img.shape)

        #kernel_m is height; kernal_n is width.
        (kernel_m, kernel_n) = kernel.shape
        (img_height, img_width) = img.shape

        k_slack_height = (kernel_m-1)/2
        k_slack_width = (kernel_n-1)/2

        img = zero_pad(img, k_slack_height, k_slack_width)

        for i in range(k_slack_height , k_slack_height+img_height):
            for j in range(k_slack_width, k_slack_width+img_width):
                G[i-k_slack_height][j-k_slack_width]= np.sum(kernel * img[i-k_slack_height:i+k_slack_height+1,j-k_slack_width:j+k_slack_width+1])
        return G

    else:
        G = np.zeros(shape = img.shape)

        (img_height, img_width, img_d) = img.shape
        (kernel_m, kernel_n) = kernel.shape

        k_slack_height = (kernel_m-1)/2
        k_slack_width = (kernel_n-1)/2

        img = zero_pad(img, k_slack_height, k_slack_width)

        for i in range(k_slack_height , k_slack_height+img_height):
            for j in range(k_slack_width, k_slack_width+img_width):
                for d in range(img_d):
                    G[i-k_slack_height][j-k_slack_width][d]= np.sum(kernel * img[i-k_slack_height:i+k_slack_height+1,j-k_slack_width:j+k_slack_width+1,d])
        return G


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = np.fliplr(np.flipud(kernel))
    return cross_correlation_2d(img, kernel)


def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    G = np.zeros(shape=(width,height))

    for w in range(-(width/2), (width/2)+1):
        for h in range(-(height/2),(height/2)+1):
            G[w+width/2][h+height/2] = np.e**(-(w**2.0 + h**2.0)/(2.0*sigma**2.0))/(2.0*np.pi*sigma**2.0)
    return G/np.sum(G)


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    low_pass_img = low_pass(img, sigma, size)
    return (img - low_pass_img)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2)
    if len(hybrid_img.shape) == 3: # if its an RGB image
        for c in range(3):
            hybrid_img[:, :, c]  /= np.amax(hybrid_img[:, :, c])
    else:
        hybrid_img /= np.amax(hybrid_img)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


