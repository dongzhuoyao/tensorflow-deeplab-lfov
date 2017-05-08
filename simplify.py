"""Run DeepLab-LargeFOV on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import glob
import cv2

from PIL import Image

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels
from deeplab_lfov.utils import single_channel_process




import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import imsave


SAVE_DIR = './rf/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


"""
img = img_as_float(astronaut()[::2, ::2])

segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=200, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')


for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
plt.savefig("slic.png")
plt.close()
"""


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("--model_weights", type=str,default="./model.ckpt-pretrained",
                        help="Path to the file with model weights.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    image = tf.placeholder("uint8", [None, None, 3])
    # Prepare image.
    img = origin_img = image
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2 )
    img = tf.cast(tf.concat([img_b, img_g, img_r],2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    deeplab = DeepLabLFOVModel()

    # Which variables to load.
    trainable = tf.trainable_variables()
    
    # Predictions.
    pred,confidence = deeplab.preds(tf.expand_dims(img, dim=0))

    print("====global variable shape check====")
    for v in tf.global_variables():
        print("{}:  {}".format(v.name, v.get_shape()))
      
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    saver = tf.train.Saver(var_list=trainable)
    load(saver, sess, args.model_weights)



    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    threshold = 0.0001

    cur_image =cv2.imread("./test/2007_008747.jpg")
    w = 226;h = 394  # xiaofangshuan

    cur_image = cv2.imread("./test/2008_000270.jpg")
    w=212;h=100#mother head

    pred_result, _img, fm8, fc8_w, confidence_cubic = sess.run([pred, origin_img, deeplab.fm8, \
                                                                deeplab.fc8_w, deeplab.confidence_cubic],
                                                               feed_dict={image: cur_image})

    origin_max_class = np.argmax(confidence_cubic[0, h, w, :], axis=0)
    origin_confidence = confidence_cubic[0, h, w, origin_max_class]
    print("origin max class: {}".format(origin_max_class))  # class 5
    print("origin max probobility: {}".format(origin_confidence))


    segments_slic = slic(cur_image, n_segments=100, compactness=10, sigma=1)
    slic_result = mark_boundaries(cur_image, segments_slic)
    imsave("slic_result.jpg", slic_result)
    max_mask = np.amax(segments_slic)

    candidate_list = []
    for i in range(max_mask+1):
        candidate_list.append(i)
    mask = np.ones_like(cur_image)

    segments_slic = np.expand_dims(segments_slic,axis=2)
    segments_slic = np.repeat(segments_slic,3,axis=2)
    while len(candidate_list):
        reduced_confidence_list = []
        print("====new iteration====")
        for i,seg_id in enumerate(candidate_list):
            tmp = np.ones_like(segments_slic)*seg_id
            tmp = -np.equal(tmp,segments_slic)
            tmp_mask = np.multiply(mask,tmp)

            tmp_img = np.multiply(tmp_mask,cur_image)

            pred_result, _img, fm8, fc8_w, confidence_cubic = sess.run([pred, origin_img, deeplab.fm8, \
                                                                deeplab.fc8_w, deeplab.confidence_cubic],
                                                               feed_dict={image: tmp_img})

            cur_max_class = np.argmax(confidence_cubic[0, h, w, :], axis=0)

            cur_confidence = confidence_cubic[0, h, w, origin_max_class]
            print("cur max class: {}".format(cur_max_class))
            print("cur max probobility: {}".format(cur_confidence))
            reduced_confidence_list.append((math.fabs(cur_confidence-origin_confidence),tmp_mask,seg_id,cur_max_class))

        reduced_confidence_list = [x for x in reduced_confidence_list if x[3]==origin_max_class and x[0] < threshold]

        # halt criterion
        if  len(reduced_confidence_list)==0:
            print("halt condition meets.break~")
            break

        #sort by confidence
        reduced_confidence_list.sort(key=lambda x: x[0])

        #mask update
        mask = np.multiply(mask,reduced_confidence_list[0][1])

        #delete from candidate
        candidate_list.remove(reduced_confidence_list[0][2])

    final_image = np.multiply(mask,cur_image)
    cv2.imwrite("simplify_result.jpg", final_image)








    
if __name__ == '__main__':
    main()
