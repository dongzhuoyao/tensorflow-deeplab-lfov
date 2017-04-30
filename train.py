"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

#SGDOptimiser,gradient multi 0.1 for every 2000 batch,weight_cay 0.05,momention 0.9.


from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
from deeplab_lfov.utils_from_resnet import decode_labels_by_batch,inv_preprocess,single_channel_process,attention_map_process

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import math
from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels

BATCH_SIZE = 16
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000000
RANDOM_SCALE = True
RESTORE_FROM = './deeplab_lfov.ckpt'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 500
SUMMARY_FREQ = 5
weight_decay = 0.0002
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = None

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")

    parser.add_argument("--summary_freq", type=int, default=SUMMARY_FREQ,
                        help="summary_freq"
                             "summary_freq")
    parser.add_argument("--summay_dir", type=str, default="./summary",
                        help="logs")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    
def load(loader, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            RANDOM_SCALE,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabLFOVModel(args.weights_path)





    # Define the loss and optimisation parameters.
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        _,hed_total_cost,predict_4d_label,cam_pre,cam_gt,confidence_map = net.loss(image_batch, label_batch,weight_decay=weight_decay)

    confidence_map_print = tf.Print(confidence_map, [tf.reduce_max(confidence_map)],'argmax(confidence_map) = ', summarize=20, first_n=100)
    cam_gt_print = tf.Print(cam_gt, [tf.reduce_max(cam_gt)], 'argmax(cam_gt) = ',
                                    summarize=20, first_n=100)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimiser = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

    #regularization
    regularization_list = [v for v in tf.global_variables() if 'block'  in v.name]
    print("====regularization_list  check====")
    for v in regularization_list:
        print("{}:  {}".format(v.name, v.get_shape()))

    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in regularization_list]) *weight_decay
    hed_total_cost = hed_total_cost+lossL2

    print("====global variable shape check====")
    for v in tf.global_variables():
        print("{}:  {}".format(v.name, v.get_shape()))

    frozen_trainalbe = [u'conv1', u'conv2', u'conv3', u'conv4', u'conv5', u'fc6', u'fc7', u'fc8']
    final_trainable = tf.global_variables()
    for f in frozen_trainalbe:
        final_trainable = [x for x in final_trainable if f not in x.name]
    print("====trainable shape check====")
    for v in final_trainable:
        print("{}:  {}".format(v.name, v.get_shape()))

    frozen_trainalbe = [u'conv1_', u'conv2_', u'conv3_', u'conv4_', u'conv5_', u'fc6_', u'fc7_', u'fc8_']
    recover_variables = []
    for f in frozen_trainalbe:
        recover_variables.extend([x for x in tf.global_variables() if f in x.name])
    print("====recover_variables shape check====")
    for v in recover_variables:
        print("{}:  {}".format(v.name, v.get_shape()))


    optim = optimiser.minimize(hed_total_cost, var_list=final_trainable)

    images_summary = tf.py_func(inv_preprocess, [image_batch, SAVE_NUM_IMAGES], tf.uint8)
    labels_summary = tf.py_func(decode_labels_by_batch, [label_batch, SAVE_NUM_IMAGES], tf.uint8)
    predict_summary = tf.py_func(decode_labels_by_batch, [predict_4d_label, SAVE_NUM_IMAGES], tf.uint8)

    gt_att_summary = tf.py_func(single_channel_process, [cam_gt, SAVE_NUM_IMAGES], tf.uint8)
    predicted_att_summary = tf.py_func(single_channel_process, [cam_pre, SAVE_NUM_IMAGES], tf.uint8)
    confidence_summay = tf.py_func(single_channel_process, [confidence_map, SAVE_NUM_IMAGES], tf.uint8)

    # define Summary
    summary_list = []
    #for var in tf.trainable_variables():
    #    summary_list.append(tf.summary.histogram(var.op.name + "/values", var))

    summary_list.append(tf.summary.histogram('cam_gt', cam_gt))
    summary_list.append(tf.summary.histogram('confidence_map', confidence_map))

    # summary
    with tf.name_scope("loss_summary"):
        summary_list.append(tf.summary.scalar("main_loss", hed_total_cost))

    with tf.name_scope("image_summary"):
        # origin_summary = tf.summary.image("origin", images_summary)
        # label_summary = tf.summary.image("label", labels_summary)
        summary_list.append(tf.summary.image('total_image',
                                             tf.concat([images_summary, labels_summary, predict_summary,gt_att_summary, predicted_att_summary], 2),
                                             max_outputs=SAVE_NUM_IMAGES))

        summary_list.append(tf.summary.image('confidence_map',
                                             tf.concat([images_summary, labels_summary,predict_summary, confidence_summay], 2),
                                             max_outputs=SAVE_NUM_IMAGES))

    merged_summary_op = tf.summary.merge_all()



    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    summary_writer = tf.summary.FileWriter(args.summay_dir, sess.graph)
    
    # Saver for storing checkpoints of the model.
    readSaver = tf.train.Saver(var_list=recover_variables, max_to_keep=40)
    writeSaver = tf.train.Saver(max_to_keep=20)
    if args.restore_from is not None:
        load(readSaver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        #get learning rate
        lr_scale = math.floor(step/4000);
        cur_lr = args.learning_rate/math.pow(10,lr_scale)
        print ("current learning rate: {}".format(cur_lr))

        _,_,loss_value, _ = sess.run([cam_gt_print,confidence_map_print,hed_total_cost, optim],feed_dict={learning_rate: cur_lr})



        if step % args.save_pred_every == 0:
            save(writeSaver, sess, args.snapshot_dir, step)
        if step % args.summary_freq == 0:
            print("write summay...")
            # generate summary for tensorboard
            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, step)

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.5f}, ({:.5f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
