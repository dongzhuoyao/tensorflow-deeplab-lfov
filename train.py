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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import math
from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels
from deeplab_lfov.utils_from_resnet import decode_labels_by_batch,inv_preprocess,single_channel_process,attention_map_process


BATCH_SIZE = 20
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
INPUT_SIZE = '321,321'
LEARNING_RATE = 1e-4
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000000
RANDOM_SCALE = True
RESTORE_FROM = './deeplab_lfov.ckpt'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10
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
    parser.add_argument("--summary_dir", type=str, default="./summary/",
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")
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
            './dataset/val.txt',
            input_size,
            RANDOM_SCALE,
            coord)
        val_image_batch, val_label_batch = reader.dequeue(args.batch_size)



    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            RANDOM_SCALE,
            coord)
        train_image_batch, train_label_batch = reader.dequeue(args.batch_size)

    is_validation = tf.placeholder(dtype=tf.bool, shape=[])
    image_batch = tf.cond(is_validation, lambda: val_image_batch, lambda: train_image_batch)
    label_batch = tf.cond(is_validation, lambda: val_label_batch, lambda: train_label_batch)

    # Create network.
    net = DeepLabLFOVModel(args.weights_path)
    # Define the loss and optimisation parameters.
    loss = net.loss(image_batch, label_batch, weight_decay=0.05)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    print("====global variable shape check====")
    for v in tf.global_variables():
        print("{}:  {}".format(v.name, v.get_shape()))

    recoverable = []
    for choosed in [u'conv1', u'conv2', u'conv3', u'conv4', u'conv5', u'fc6', u'fc7', u'fc8']:
        for tmp in tf.global_variables():
            if choosed in tmp.name:
                recoverable.append(tmp)

    print("====recoverable shape check====")
    for v in recoverable:
        print("{}:  {}".format(v.name, v.get_shape()))

    stage_var = []
    for choosed in [u'stage']:
        for tmp in tf.global_variables():
            if choosed in tmp.name:
                stage_var.append(tmp)

    print("====stage_var shape check====")
    for v in stage_var:
        print("{}:  {}".format(v.name, v.get_shape()))





    optim_1 = tf.train.MomentumOptimizer(learning_rate=learning_rate*0.1, momentum=0.9).minimize(loss, var_list=recoverable)
    optim_2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, var_list=stage_var)
    optim = tf.group(optim_1,optim_2)


    pred = net.preds(image_batch)

    images_summary = tf.py_func(inv_preprocess, [image_batch, SAVE_NUM_IMAGES], tf.uint8)
    labels_summary = tf.py_func(decode_labels_by_batch, [label_batch, SAVE_NUM_IMAGES], tf.uint8)
    predict_summary = tf.py_func(decode_labels_by_batch, [pred, SAVE_NUM_IMAGES], tf.uint8)


    # define Summary
    summary_list = []
    for var in tf.trainable_variables():
        summary_list.append(tf.summary.histogram(var.op.name + "/values", var))

    # summary
    with tf.name_scope("loss_summary"):
        summary_list.append(tf.summary.scalar("main_loss", loss))

    with tf.name_scope("image_summary"):
        # origin_summary = tf.summary.image("origin", images_summary)
        # label_summary = tf.summary.image("label", labels_summary)
        summary_list.append(tf.summary.image('total_image',
                                             tf.concat([images_summary, labels_summary, predict_summary], 2),
                                             max_outputs=SAVE_NUM_IMAGES))


    merged_summary_op = tf.summary.merge(summary_list)



    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)


    
    # Saver for storing checkpoints of the model.
    readSaver = tf.train.Saver(var_list=recoverable, max_to_keep=40)
    writeSaver = tf.train.Saver(max_to_keep=20)

    if args.restore_from is not None:
        load(readSaver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    # Iterate over training steps.
    for step in range(1,args.num_steps):
        start_time = time.time()
        #get learning rate
        lr_scale = math.floor(step/4000);
        cur_lr = args.learning_rate/math.pow(10,lr_scale)
        print ("current learning rate: {}".format(cur_lr))


        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, _ = sess.run([loss, image_batch, label_batch, pred, optim],feed_dict={learning_rate:cur_lr,is_validation:False})
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0]))
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            save(writeSaver, sess, args.snapshot_dir, step)

            print("write summay...")
            # generate summary for tensorboard
            summary_str = sess.run(merged_summary_op,feed_dict={learning_rate: cur_lr,is_validation:True})
            summary_writer.add_summary(summary_str, step)

            val_loss_value, images, labels, preds, _ = sess.run([loss, image_batch, label_batch, pred, optim],
                                                            feed_dict={learning_rate: cur_lr,is_validation:True})
            print('step {:d} \t validation loss = {:.3f}, ({:.3f} sec/step)'.format(step, val_loss_value, duration))


        else:
            loss_value, _ = sess.run([loss, optim],feed_dict={learning_rate:cur_lr,is_validation:False})
        duration = time.time() - start_time

        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
