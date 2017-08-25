import load_alexnet
import student
import numpy as np
import pytest

# relative error threshold
THRESH = 1e-2


#### LOAD ALEXNET ####
# Fix the PYTHONPATH before importing caffe
import os, sys
caffe_root = os.path.expanduser('~/caffe/')
sys.path.insert(0, caffe_root + 'python')
import caffe
net, transformer = load_alexnet.load_alexnet(caffe_root, gpu=False)


def _load(fname):
    """ Helper to load compressed .npz arrays """
    return np.load(fname)['arr_0']


def relative_error(expected, actual, epsilon=1e-5):
    """ Mean relative error between two arrays """
    return np.mean(
        np.abs(expected - actual) /
        np.maximum(np.abs(actual), epsilon)
    )


def test_compute_dscore_dimage():
    data = _load('expected/data.npz')
    expected = _load('expected/compute_dscore_dimage-grad.npz')

    net.blobs['data'].data[0, ...] = data
    net.forward()
    actual = student.compute_dscore_dimage(net, data, 254)

    # relative error must be small
    assert relative_error(expected, actual) < THRESH


def test_normalized_sgd_with_momentum_update():
    data = _load('expected/data.npz')
    grad = _load('expected/normalized_sgd_with_momentum_update-grad.npz')
    velocity = _load('expected/normalized_sgd_with_momentum_update-velocity.npz')
    expected_data = _load('expected/normalized_sgd_with_momentum_update-new_data.npz')
    expected_velocity = _load('expected/normalized_sgd_with_momentum_update-new_velocity.npz')

    momentum = 0.99
    learning_rate = 100

    new_data, new_velocity = student.normalized_sgd_with_momentum_update(
        data, grad, velocity, momentum, learning_rate)

    assert relative_error(expected_data, new_data) < THRESH
    assert relative_error(expected_velocity, new_velocity) < THRESH


def test_fooling_image_gradient():
    data = _load('expected/data.npz')
    expected = _load('expected/fooling_image_gradient-grad.npz')
    orig_data = _load('expected/fooling_image_gradient-orig_data.npz')

    net.blobs['data'].data[0, ...] = data
    net.forward()

    target_class = 113
    regularization = 1e-3

    actual = student.fooling_image_gradient(
        net, orig_data, data, target_class, regularization)

    # relative error must be small
    assert relative_error(expected, actual) < THRESH


def test_class_visualization_gradient():
    cur_data = _load('expected/class_visualization_gradient-cur_data.npz')
    expected = _load('expected/class_visualization_gradient-grad.npz')

    net.blobs['data'].data[0, ...] = cur_data
    net.forward()

    target_class = 234
    regularization = 1e-3

    actual = student.class_visualization_gradient(
        net, cur_data, target_class, regularization)

    # relative error must be small
    assert relative_error(expected, actual) < THRESH


def test_feature_inversion_gradient():
    blob_name = 'conv3'
    regularization = 2e-3

    target_feat = _load('expected/feature_inversion_gradient-target_feat.npz')
    cur_data = _load('expected/feature_inversion_gradient-cur_data.npz')
    expected = _load('expected/feature_inversion_gradient-grad.npz')

    net.blobs['data'].data[0, ...] = cur_data
    net.forward(end=blob_name)

    actual = student.feature_inversion_gradient(
        net, cur_data, blob_name, target_feat, regularization)

    # relative error must be small
    assert relative_error(expected, actual) < THRESH
