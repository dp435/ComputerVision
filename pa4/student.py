# Please place imports here.
# BEGIN IMPORTS
import numpy as np
import numpy.ma as ma
import cv2
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Columns are normalized and are to be
                  interpreted as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.  Images are height x width x channels arrays, all
                  with identical dimensions.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # Calculate dimensions for inputs & outputs.
    (height, width, channel) = images[0].shape
    N = len(images)

    # Initialize arrays for albedo & normals; dtype='float32'.
    albedo = np.zeros((height,width,channel),dtype='float32')
    normals = np.zeros((height,width,3),dtype='float32')

    # Accumulator to hold G of each RGB color channel.
    G = np.zeros((height*width,3,channel),dtype='float32')

    for ch in xrange(channel):
        # Stack images in a (pixel x num_images) array.
        I = np.zeros((height*width, N),dtype='float32')
        for img_idx in xrange(N):
            I[:,img_idx] = images[img_idx][:,:,ch].reshape(height*width,)

        # Store G_RGB in G_Accumulator.
        G[:,:,ch] = I.dot(lights.T).dot(np.linalg.inv(lights.dot(lights.T)))

    # Calculate the albedo.
    G_norm = np.linalg.norm(G,axis=1)
    G_norm = G_norm[:,np.newaxis]
    G_norm[G_norm < 1.0e-7] = 0.0
    albedo = G_norm.reshape(height,width,channel)

    # Calculate G_grey matrix.
    G_grey = (np.sum(G, axis=2)/3.0)

    # Calculate ||G_grey||; set entries less than 1e-7 to 0.
    G_grey_norm = np.linalg.norm(G_grey, axis=1).astype(dtype='float32')
    G_grey_norm = G_grey_norm[:,np.newaxis]
    G_grey_norm[G_grey_norm < 1e-7] = 0.0

    # Mask for ||G_grey||; True for index containing 0-values.
    mask = ma.equal(G_grey_norm, 0.0)

    # Replace 0-values with 1.0 temporarily in ||G_grey||.
    G_grey_norm[mask] = 1.0

    # Calculate normals.
    N_matrix = (G_grey/G_grey_norm)
    N_matrix[mask] = 0.0
    normals = N_matrix.reshape(height,width,3)

    return albedo, normals


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and
    scipy.ndimage.filters.gaussian_filter are prohibited.  You must implement
    the separable kernel.  However, you may use functions such as cv2.filter2D
    or scipy.ndimage.filters.correlate to do the actual
    correlation / convolution.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) [x channels] image of type
                float32.
    """
    K = (1.0/16.0)*np.array([[1,4,6,4,1]])
    down = cv2.filter2D(image, -1, K, borderType=cv2.BORDER_REFLECT_101)
    K_transposed = K.T
    down = cv2.filter2D(down, -1, K_transposed, borderType=cv2.BORDER_REFLECT_101)

    down = down[::2,::2]

    return down


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        up -- 2 height x 2 width [x channels] image of type float32.
    """

    if (image.ndim == 2):
        height,width = image.shape
        up = np.zeros((height*2.0, width*2.0))
    else:
        height,width,channel = image.shape
        up = np.zeros((height*2.0, width*2.0, channel))

    up[::2,::2] = image

    K = (1.0/8.0)*np.array([[1,4,6,4,1]])
    up = cv2.filter2D(up, -1, K, borderType=cv2.BORDER_REFLECT_101)
    K_transposed = K.T
    up = cv2.filter2D(up, -1, K_transposed, borderType=cv2.BORDER_REFLECT_101)

    return up


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    If the point has a depth < 1e-7 from the camera or is located behind the
    camera, then set the projection to [np.nan, np.nan].

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height,width,channel = points.shape
    projections = np.zeros((height,width,2))
    ARt = K.dot(Rt)
    points = np.c_[points, np.ones((height,width,1))]

    reshaped_points = points.reshape((height*width, 4))

    projected_pts = ARt.dot(reshaped_points.T)

    p_channel, p = projected_pts.shape
    threshold = np.ones((p,))*(1.0e-7)

    less_than_threshold = np.less(projected_pts[2],threshold)

    projected_pts[2][less_than_threshold] = 1.0

    projected_pts = projected_pts/projected_pts[2]

    projected_pts[0][less_than_threshold] = np.nan
    projected_pts[1][less_than_threshold] = np.nan

    projections = projected_pts[:2].T.reshape((height,width,2))

    return projections


def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z, 1) - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    corners = np.array([np.array([0,0,1]),np.array([width,0,1]),np.array([0,height,1]),np.array([width,height,1])])

    K_inv = np.linalg.inv(K)

    corners = K_inv.dot(corners.T)

    corners = depth * (corners/corners[2,:])

    Rt_inv = np.linalg.inv(np.vstack((Rt,np.array([0,0,0,1]))))

    R_prime = Rt_inv[:3,:3]
    Rt_prime = Rt_inv[:3,3]

    p = R_prime.dot(corners) + Rt_prime[:,np.newaxis]
    p = p.T.reshape(2,2,3)
    
    return p


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the offset of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches of shape channels x height x width (e.g. 3 x ncc_size x ncc_size)
    are to be flattened into vectors with the default numpy row major order.
    For example, given the following 2 (channels) x 2 (height) x 2 (width)
    patch, here is how the output vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.  ncc_size
                    will always be odd.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height,width,channel = image.shape
    patches = np.zeros((height,width, channel, ncc_size * ncc_size),dtype='float32')
    normalized = np.zeros((height,width,channel*ncc_size**2),dtype='float32')
    offset = int(np.floor(ncc_size/2.0))

    for h in xrange(offset,height-offset):
        for w in xrange(offset, width-offset):
            patches[h,w] = np.transpose(image[h-offset:h+offset+1,w-offset:w+offset+1],axes=(2,0,1)).reshape(channel,ncc_size**2)

    patches = patches - np.mean(patches,axis=3)[:,:,:,np.newaxis]

    norm = np.linalg.norm(patches, axis=(2,3))

    mask = ma.less(norm, 1.0e-6)

    norm[mask] = 1.0
    patches = patches / norm[:,:,np.newaxis,np.newaxis]

    patches[mask] = 0.0

    normalized = patches.reshape(height,width,channel*ncc_size**2)

    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    return np.sum((image1*image2), axis=2)
