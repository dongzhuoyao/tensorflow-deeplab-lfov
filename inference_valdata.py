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


from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels

SAVE_DIR = './output/'
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("--txt_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("--model_weights", type=str,
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

    img_path = tf.placeholder(tf.string)
    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2 )
    img = tf.cast(tf.concat([img_b, img_g, img_r],2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabLFOVModel()

    # Which variables to load.
    trainable = tf.trainable_variables()
    
    # Predictions.
    pred = net.preds(tf.expand_dims(img, dim=0))
      
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    saver = tf.train.Saver(var_list=trainable)
    load(saver, sess, args.model_weights)

    f = open(args.txt_path,"r")
    test_img_list = f.readlines()
    test_img_list = [x.strip() for x in test_img_list]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    for i,line in enumerate(test_img_list):
        tmp = line.split(" ")
        tmp[0] = tmp[0][1:]#delete first /
        tmp[1] = tmp[1][1:]
        image_path = os.path.join(args.img_path,tmp[0])
        label_path = os.path.join(args.img_path,tmp[1])
        image = cv2.imread(image_path)
        label = cv2.imread(label_path,0)

        fig, axes = plt.subplots(2, 3, figsize=(15,10))
        fig.patch.set_visible(False)#http://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame



        pred_result = sess.run([pred], feed_dict={img_path:image_path})
        #pre_result is a list!!!
        pred_result = np.array(pred_result)[0,0, :, :, 0]
        print("np.array(pred_result) shape: {}".format(pred_result.shape))

        wrong = np.not_equal(pred_result.astype(np.uint8),label.astype(np.uint8))*255



        img_name = os.path.basename(image_path)
        img_name = img_name.replace("jpg", "png")

        axes.flat[0].axis('off')
        axes.flat[0].set_title('data')
        axes.flat[0].imshow(image.astype(np.uint8))

        axes.flat[1].axis('off')
        axes.flat[1].set_title('mask')
        axes.flat[1].imshow(decode_labels(label))

        axes.flat[2].axis('off')
        axes.flat[2].set_title('pred')
        axes.flat[2].imshow(decode_labels(pred_result))

        axes.flat[3].axis('off')
        axes.flat[3].set_title('wrong')
        axes.flat[3].imshow(wrong.astype(np.uint8))


        print('The output file has been saved to {}'.format(os.path.join(args.save_dir,img_name)))
        plt.savefig(os.path.join(args.save_dir,img_name))
        plt.close(fig)

    
if __name__ == '__main__':
    main()
