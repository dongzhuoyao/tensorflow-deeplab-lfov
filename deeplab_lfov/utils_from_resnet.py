from PIL import Image
import numpy as np
import tensorflow as tf

n_classes = 21
# colour map
label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# colour map
raw_label_colours = [(0, 0, 0)
                     # 0=background
    , (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)
                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (6, 6, 6), (7, 7, 7), (8, 8, 8), (9, 9, 9), (10, 10, 10)
                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (11, 11, 11), (12, 12, 12), (13, 13, 13), (14, 14, 14), (15, 15, 15)
                     # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (16, 16, 16), (17, 17, 17), (18, 18, 18), (19, 19, 19), (20, 20, 20)]
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# image mean
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def decode_labels_by_batch(mask, num_images=1, raw_label_output=False):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < n_classes:
                    if raw_label_output:
                        pixels[k_, j_] = raw_label_colours[k]
                    else:
                        pixels[k_, j_] = label_colours[k]

        outputs[i] = np.array(img)
    return outputs


def prepare_label(input_batch, new_size, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch,
                                                       new_size)  # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3])  # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch


def inv_preprocess(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
    return outputs


def single_channel_process(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape#c=1,because attention map has only one channel
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        tmp = (imgs[i])[:, :, ::-1].flatten()
        bins = np.array([0.0, 0.95, 1.0])
        tmp = np.digitize(tmp, bins)
        tmp = np.multiply(tmp,255.0 / tmp.max())
        tmp = tmp.reshape((h,w)).astype(np.uint8)

        outputs[i, :, :, 0] = tmp
        outputs[i, :, :, 1] = tmp
        outputs[i, :, :, 2] = tmp
    return outputs


def attention_map_process(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape#c=1,because attention map has only one channel
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        outputs[i, :,  :,0] = (255*imgs[i])[:, :, ::-1].astype(np.uint8).reshape((h,w))
        outputs[i, :, :, 1] = (255 * imgs[i])[:, :, ::-1].astype(np.uint8).reshape((h,w))
        outputs[i, :, :, 2] = (255 * imgs[i])[:, :, ::-1].astype(np.uint8).reshape((h,w))
    return outputs