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

SAVE_DIR = './rf/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

#cur_img_path = "./test/2008_000270.jpg"
#cur_img_path = "./test/2008_000234.jpg"
cur_img_path = "./test/2007_008747.jpg"
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

    cur_image =cv2.imread("./test/2007_008747.jpg")
    img_w,img_h,img_c = cur_image.shape

    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    w = 226;h = 394  # xiaofangshuan
    #fig.patch.set_visible(False)#http://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame

    pred_result, _img, fm8, fc8_w, confidence_cubic = sess.run([pred, origin_img, deeplab.fm8, \
                                                                deeplab.fc8_w, deeplab.confidence_cubic],
                                                               feed_dict={image: cur_image})

    origin_max_class = np.argmax(confidence_cubic[0, h, w, :], axis=0)
    origin_confidence = confidence_cubic[0, h, w, origin_max_class]
    print("origin max class: {}".format(origin_max_class))  # class 15:person
    print("origin max probobility: {}".format(origin_confidence))


    noise_np = np.random.rand(11,11,3)*255
    candidate_list = []
    window_size = 11
    step = 5
    for i in range(0,img_w-window_size,step):
        for j in range(0,img_h-window_size,step):
            tmp_img = np.copy(cur_image)
            tmp_img[i:i+window_size,j:j+window_size,:] = noise_np
            #cv2.imwrite("rrr.jpg",tmp_img)

            pred_result,_img,fm8,fc8_w,confidence_cubic = sess.run([pred,origin_img,deeplab.fm8,\
            deeplab.fc8_w,deeplab.confidence_cubic], feed_dict={image:tmp_img})


            print("max probobility: {}".format(confidence_cubic[0,h,w,origin_max_class]))
            candidate_list.append((i,j,math.fabs(origin_confidence-confidence_cubic[0,h,w,origin_max_class])))


    candidate_list.sort(key=lambda x: x[2], reverse=True)

    result = np.zeros_like(cur_image,dtype=np.float64)
    for m in range(100):
        i,j,activation = candidate_list[m]
        result[i:i+window_size,j+window_size,:] = activation

    result *= 255.0 / result.max()

    cv2.imwrite("rf_result.jpg",result)






    #axes.flat[0].set_title('mask')
    #axes.flat[0].imshow(final_fm)

    #axes.flat[1].set_title('src')
    #axes.flat[1].imshow(_img)


    #pre_result is a list!!!
    #pred_result = np.array(pred_result)[0, :, :, 0]

    #confidence_result = np.array(confidence_result)[0, :, :, 0]
    print("np.array(pred_result) shape: {}".format(pred_result.shape))
    plt.savefig("TAT.png")
    plt.close(fig)




    
if __name__ == '__main__':
    main()
