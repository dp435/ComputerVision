import os, sys,  math, random, subprocess
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, Image, display, HTML
from google.protobuf import text_format
import PIL.Image

def load_alexnet(caffe_root, gpu=False):
    """
    Load the AlexNet model in pycaffe, as well as the transformer which can
    convert images to/from the format required by the first layer ("data").

    :returns: (net, transformer)
    """

    # Note: assumes that the python path is configured
    import caffe

    # Download AlexNet
    fn_model = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
    print 'looking for file: ', fn_model
    if os.path.isfile(caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'):
        print 'AlexNet found.'
    else:
        print 'Downloading pre-trained CaffeNet model...'
        subprocess.call(['python', caffe_root + 'scripts/download_model_binary.py',
        caffe_root + 'models/bvlc_alexnet'])

    # 2. Load net and set up input preprocessing
    if gpu:
        caffe.set_mode_gpu()
    else:
        # * Set Caffe to CPU mode and load the net from disk.
        caffe.set_mode_cpu()

    model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
    model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(model_def).read(), model)
    model.force_backward = True
    model.layer[0].input_param.shape[0].dim[0] = 1  # set batchsize to 1
    open('patched_alexnet.prototxt', 'w').write(str(model))

    net = caffe.Net('patched_alexnet.prototxt', # defines the structure of the model
                    model_weights,              # contains the trained weights
                    caffe.TEST)              # use test mode (e.g., don't perform dropout)


    # * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this,
    # but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
    # Our default AlexNet is configured to take images in BGR format. Values are expected to start
    # in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them.
    # In addition, the channel dimension is expected as the first (outermost) dimension.

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(1,         # batch size
                            3,         # 3-channel (BGR) images
                            227, 227)  # image size is 227x227
    net.reshape()


    print 'Loaded net', net

    return net, transformer
