import numpy as np
import scipy.signal as sig
from scipy.ndimage.filters import convolve
import imageio
import skimage.color as skimage
import math
from matplotlib import pyplot as plt
import os

GRAY_SCALE = 1
RGB = 2
EXPEND_NORM = 2
RGB_DIM = 3
EXAMPLE_SIZE = 3
NORMALIZED = 255
MIN_SIZE = (16 * 16)
REDUCE_CONVOLVE = np.array([1.0, 1.0])
NORM_REDUCE_BLUR = 1 / 4


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk(could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the
        output should be a grayscale image(1) or an RGB image(2).
    :return: An image, represented by a matrix of type np.float64 with intensities.
    """
    image = imageio.imread(filename)

    # checks if the image is already from type float64
    if not isinstance(image, np.float64):
        image.astype(np.float64)
        image = image / NORMALIZED

    # checks if the output image should be grayscale
    if representation == GRAY_SCALE:
        image = skimage.rgb2gray(image)
    return image


def calculate_filter_vec(filter_size, norm):
    """
    This function calculates the filter vector- used for the pyramid construction
    :param filter_size: the size of the gaussian filter to be used in constructing
        the pyramid filter
    :param norm: the blur normalization
    :return: the filter vector
    """
    if filter_size == 1:
        return [[1]]
    filter_vec = REDUCE_CONVOLVE
    for i in range(filter_size - 2):
        filter_vec = np.array(sig.convolve(filter_vec, REDUCE_CONVOLVE))
    power = (filter_size - 1) / 2
    filter_vec *= math.pow(norm, power)
    return filter_vec


def reduce(im, filter_vec):
    """
    This function operates the reduce algorithm
    :param im: the image to operate the reduce on
    :param filter_vec: the vector to convolve with the image in order to blur
    :return: the image after reduce
    """
    # blur image:
    layer_blur = blur(im, filter_vec)
    # sub sample the image:
    layer_blur_sample = layer_blur[::2, 1::2]
    return layer_blur_sample


def blur(im, filter_vec):
    """
    This function operates a blur on a given image
    :param filter_vec: the vector to convolve with the image in order to blur
    :param im: the image to operate the blur on
    :return: a blurred image
    """
    layer = convolve(im, filter_vec)
    layer_blur = convolve(layer, filter_vec.T)
    return layer_blur


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function construct a gaussian pyramid on a given image
    :param im: a greyscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter to be used in constructing
        the pyramid filter
    :return: pyr - the result pyramid
            filter_vec - row vector used for the pyramid construction
    """
    filter_vec = np.reshape(calculate_filter_vec(filter_size, NORM_REDUCE_BLUR),
                            (1, filter_size))

    pyr = [im]
    blur_im = im
    i = 1
    while blur_im.size > MIN_SIZE and i < max_levels:
        layer_blur_sample = reduce(blur_im, filter_vec)
        pyr.append(layer_blur_sample)
        blur_im = layer_blur_sample
        i += 1
    return pyr, filter_vec


def expend(im, filter_vec):
    """
    This function operates the expend algorithm
    :param im: the image to operate the expend on
    :param filter_vec: the vector to convolve with the image in order to blur
    :return: the image after expend
    """
    pad_arr = np.zeros((len(im) * EXPEND_NORM, len(im[0]) * EXPEND_NORM))
    pad_arr[::2, 1::2] = im
    pad_arr_blur = blur(pad_arr, EXPEND_NORM * filter_vec)
    return pad_arr_blur


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function construct a laplacian pyramid on a given image
    :param im: a greyscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter to be used in constructing
        the pyramid filter
    :return: pyr - the result pyramid
            filter_vec - row vector used for the pyramid construction
    """
    pyr = []
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    i = 1
    if max_levels == 1:
        pyr.append(gaussian_pyr[0])
        return pyr, filter_vec
    im_g = gaussian_pyr[0] - expend(gaussian_pyr[1], filter_vec)
    pyr.append(im_g)
    while im_g.size > MIN_SIZE and i < len(gaussian_pyr) - 1:
        im_g = gaussian_pyr[i] - expend(gaussian_pyr[i + 1], filter_vec)
        pyr.append(im_g)
        i += 1
    pyr.append(gaussian_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lypr, filter_vec, coeff):
    """
    This function reconstruct an image from its laplacian pyramid
    :param lypr: the laplacian pyramid of the image
    :param filter_vec: the vector to convolve with the image in order to expand
    :param coeff: a list of corresponding coefficient to multiply the image
    :return: the original image reconstructed from the laplacian pyramid
    """
    orig_image = lypr[-1] * coeff[-1]
    for i in range(len(lypr) - 2, -1, -1):
        orig_image = expend(orig_image, filter_vec) + lypr[i] * coeff[i]
    return orig_image


def render_pyramid(pyr, levels):
    """
    This function returns a single black image in which the pyramid levels of the
    given pyramid are stocked horizontally
    :param pyr: the pyramid to display (gaussian or laplacian)
    :param levels: the number of levels to present in the result
    :return: a single black image
    """
    length = len(pyr[0])
    wight = 0
    for i in range(levels):
        wight += len(pyr[i][0])
    res = np.zeros((length, wight))
    return res


def display_pyramid(pyr, levels):
    """
    This function displays the stacked pyramids image
    :param pyr: the pyramid to display (gaussian or laplacian)
    :param levels: the number of levels to present in the result
    """
    res = render_pyramid(pyr, levels)
    cur_col = 0
    for i in range(levels):
        layer = pyr[i]

        # normalize the values to [0,1] rang if necessary
        if np.amax(layer) > 1:
            layer /= np.amax(layer)
        if np.amin(layer) < 0:
            layer -= np.amin(layer)
            if np.amax(layer) > 1:
                layer /= np.amax(layer)

        row = len(layer)
        col = len(layer[0])
        res[:row, cur_col:cur_col + col] = layer
        cur_col += col
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    This function operates image blending on im1 and im2 using a given mask
    :param im1: a grayscale image to be blended
    :param im2: a grayscale image to be blended
    :param mask: a boolean mask representing which parts of im1 and im2 should appear
    in the resulting im_blend
    :param max_levels: the max_level parameter used while generating the gaussian and
    laplacian pyramids
    :param filter_size_im: the size of the gaussian which defining the filer used in
    the construction of the laplacian pyramids of im1 and im2
    :param filter_size_mask: the size of the gaussian which defining the filer used
    in the construction of the gaussian pyramids of mask
    :return: the blended image of im1 and im2 in gray scale
    """
    L1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, filter_vec3 = build_gaussian_pyramid(mask.astype(np.float64), max_levels,
                                             filter_size_mask)
    Lout = [None] * len(L1)
    for k in range(len(L1)):
        Lout[k] = Gm[k] * L1[k] + (1 - Gm[k]) * L2[k]
    im_blend = laplacian_to_image(Lout, filter_vec1, [1] * max_levels)
    return np.clip(im_blend, 0, 1)


def color_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    This function operates the pyramid_blending function on each color channel to
    present an RGB blended image
    :param im1: an RGB image to be blended
    :param im2: an RGB image to be blended
    :param mask: a boolean mask representing which parts of im1 and im2 should appear
    in the resulting im_blend
    :param max_levels: the max_level parameter used while generating the gaussian and
    laplacian pyramids
    :param filter_size_im: the size of the gaussian which defining the filer used in
    the construction of the laplacian pyramids of im1 and im2
    :param filter_size_mask: the size of the gaussian which defining the filer used
    in the construction of the gaussian pyramids of mask
    :return: the blended image of im1 and im2 in RGB
    """
    im_blend = np.zeros(im1.shape)
    for i in range(RGB_DIM):
        im_blend[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask,
                                             max_levels,
                                             filter_size_im, filter_size_mask)
    return im_blend


def relpath(filename):
    """
    This function returns a relative path for a given file
    :param filename: the file to return its relative path
    :return: relative path for the given file
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    This function performs pyramid blending on two images and a mask I found nice
    :return: the 2 RGB images, the boolean mask and the blended image
    """
    im1 = read_image(relpath("externals/ariana.jpg"), RGB).astype(np.float64)
    im2 = read_image(relpath("externals/dragon.jpg"), RGB).astype(np.float64)
    mask = read_image(relpath("externals/ariana_mask.jpg"), GRAY_SCALE)
    mask = np.round(mask).astype(np.bool)
    blend = color_blending(im1, im2, mask, EXAMPLE_SIZE, EXAMPLE_SIZE,
                           EXAMPLE_SIZE).astype(np.float64)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[1, 0].imshow(mask, cmap=plt.cm.gray)
    ax[1, 1].imshow(blend)
    plt.show()
    return im1, im2, mask, blend


def blending_example2():
    """
    This function performs pyramid blending on two images and a mask I found nice
    :return: the 2 RGB images, the boolean mask and the blended image
    """
    im1 = read_image(relpath("externals/beyonce.jpg"), RGB).astype(np.float64)
    im2 = read_image(relpath("externals/throne.jpg"), RGB).astype(np.float64)
    mask = read_image(relpath("externals/beyonce_mask.jpg"), GRAY_SCALE)
    mask = np.round(mask).astype(np.bool)
    blend = color_blending(im1, im2, mask, EXAMPLE_SIZE, EXAMPLE_SIZE,
                           EXAMPLE_SIZE).astype(np.float64)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[1, 0].imshow(mask, cmap=plt.cm.gray)
    ax[1, 1].imshow(blend)
    plt.show()
    return im1, im2, mask, blend

