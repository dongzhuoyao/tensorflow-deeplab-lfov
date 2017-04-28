"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels
from deeplab_lfov.utils_from_resnet import decode_labels_by_batch,inv_preprocess,single_channel_process,attention_map_process

BATCH_SIZE = 16
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
INPUT_SIZE = '321,321'
LEARNING_RATE = 1e-4
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000000
RANDOM_SCALE = True
RESTORE_FROM = './model.ckpt-pretrained'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 4

SAVE_PRED_EVERY = 100
SUMMARY_FREQ = 100


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

    parser.add_argument("--attentionbranch_restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")

    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")

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

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

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
    net = DeepLabLFOVModel()


    # Define the loss and optimisation parameters.
    main_loss_1, pre_upscaled_1, gt_attention_map,predicted_attention_feat,predict_3d_1= net.loss(image_batch, label_batch)

    main_loss = main_loss_1


    learning_rate = tf.placeholder(tf.float32, shape=[])
    att_learning_rate = tf.placeholder(tf.float32, shape=[])
    trainable = tf.trainable_variables()

    frozen_trainalbe = [u'conv1',u'conv2',u'conv3',u'conv4']
    final_trainable = trainable
    for f in frozen_trainalbe:
        final_trainable = [x for x in final_trainable if f not in x.name]
    print("====final_trainable shape check====")
    for v in final_trainable:
        print("{}:  {}".format(v.name, v.get_shape()))



    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)



    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    images_summary = tf.py_func(inv_preprocess, [image_batch, SAVE_NUM_IMAGES], tf.uint8)
    labels_summary = tf.py_func(decode_labels_by_batch, [label_batch, SAVE_NUM_IMAGES], tf.uint8)
    preds_1_summary = tf.py_func(decode_labels_by_batch, [pre_upscaled_1, SAVE_NUM_IMAGES], tf.uint8)

    gt_att_summary = tf.py_func(single_channel_process, [gt_attention_map, SAVE_NUM_IMAGES], tf.uint8)
    predicted_att_summary = tf.py_func(single_channel_process, [predicted_attention_feat, SAVE_NUM_IMAGES], tf.uint8)

    predict_3d_1_summary = tf.py_func(single_channel_process, [predict_3d_1, SAVE_NUM_IMAGES], tf.uint8)


    # define Summary
    summary_list = []
    for var in tf.trainable_variables():
        summary_list.append(tf.summary.histogram(var.op.name + "/values", var))

    #summary
    with tf.name_scope("loss_summary"):
        summary_list.append(tf.summary.scalar("main_loss", main_loss))

    with tf.name_scope("image_summary"):
        #origin_summary = tf.summary.image("origin", images_summary)
        #label_summary = tf.summary.image("label", labels_summary)
        summary_list.append(tf.summary.image('total_image',
                         tf.concat([images_summary, labels_summary, preds_1_summary,gt_att_summary,predicted_att_summary ], 2),
                         max_outputs=SAVE_NUM_IMAGES))

        summary_list.append(tf.summary.image('confidence_map',
                                         tf.concat([images_summary, labels_summary,predict_3d_1_summary], 2),
                                         max_outputs=SAVE_NUM_IMAGES))


    merged_summary_op = tf.summary.merge(summary_list)


    summary_writer = tf.summary.FileWriter(args.summay_dir, sess.graph)


    var_to_be_trained = []
    var_to_be_trained.extend([x for x in tf.global_variables() if u'conv1' not in x.name])
    var_to_be_trained.extend([x for x in tf.global_variables() if u'conv2' not in x.name])
    var_to_be_trained.extend([x for x in tf.global_variables() if u'conv3' not in x.name])
    var_to_be_trained.extend([x for x in tf.global_variables() if u'conv4' not in x.name])
    var_to_be_trained.extend([x for x in tf.global_variables() if u'aggregated_feat' not in x.name])

    print("====trainable_variables shape check====")
    for v in var_to_be_trained:
        print("{}:  {}".format(v.name, v.get_shape()))
    optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(main_loss,
                                                                                           var_list=var_to_be_trained)

    # check the shape
    print("====global_variables shape check====")
    for v in tf.global_variables():
        print("{}:  {}".format(v.name, v.get_shape()))


    # don't need initiate "filter_of_attention_map"!!!
    var_to_be_restored =  [x for x in trainable if u'filter_of_attention_map' not in x.name]
    var_to_be_restored = [x for x in var_to_be_restored if u'aggregated_feat' not in x.name]
    var_to_be_restored = [x for x in var_to_be_restored if u'fc6_layer_extra' not in x.name]


    uninitialized_vars =[]
    uninitialized_vars.extend([x for x in tf.global_variables() if u'filter_of_attention_map' in x.name])
    uninitialized_vars.extend([x for x in tf.global_variables() if u'aggregated_feat' in  x.name])
    uninitialized_vars.extend([x for x in tf.global_variables() if u'Variable' in x.name])
    uninitialized_vars.extend([x for x in tf.global_variables() if u'Momentum' in x.name])
    uninitialized_vars.extend([x for x in tf.global_variables() if u'fc6_layer_extra' in x.name])


    # Saver for storing checkpoints of the model.
    print("====var_to_be_restored shape check====")
    for tmp in var_to_be_restored:
        print("variable name: {},shape: {}, type: {}".format(tmp.name,tmp.get_shape(),type(tmp.name)))

    # Saver for storing checkpoints of the model.
    print("====uninitialized_vars shape check====")
    for tmp in uninitialized_vars:
        print("variable name: {},shape: {}, type: {}".format(tmp.name, tmp.get_shape(), type(tmp.name)))

    readSaver = tf.train.Saver(var_list=var_to_be_restored)
    writeSaver = tf.train.Saver( max_to_keep=40)
    if args.restore_from is not None:
       load(readSaver, sess, args.restore_from)

    att_var_to_be_restored = []
    att_var_to_be_restored.extend([x for x in tf.global_variables() if u'aggregated_feat' in x.name])

    attention_branch_readSaver = tf.train.Saver(var_list=att_var_to_be_restored)
    load(attention_branch_readSaver, sess, args.attentionbranch_restore_from)


    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    sess.run(init_new_vars_op)

    print("====varaible initialize status check====")
    init_flag = sess.run(
        tf.stack([tf.is_variable_initialized(v) for v in tf.global_variables()]))
    for v, flag in zip(tf.global_variables(), init_flag):
        if not flag:
            print ("====warning====")
        print("name: {},  shape: {}, is_variable_initialized:{}".format(v.name, v.get_shape(), flag))

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    print("parameter_count =", sess.run(parameter_count))

    summary_str = sess.run(merged_summary_op)
    summary_writer.add_summary(summary_str)

    # Iterate over training steps.
    for step in range(1,args.num_steps):
        start_time = time.time()

        # get learning rate
        lr_scale = math.floor(step / 2000);
        cur_lr = args.learning_rate / math.pow(2, lr_scale)
        print("current learning rate: {}".format(cur_lr))


        _,_main_loss,\
        _main_loss_1, preds_result_value, \
         images, labels\
        = sess.run([optim,main_loss,main_loss_1, pre_upscaled_1,image_batch,label_batch],feed_dict={learning_rate:cur_lr,att_learning_rate:cur_lr})

        print('step {:d}, main_loss: {:.5f}, loss 1: {:.5f},'.format(step,_main_loss,_main_loss_1,))

        if step % args.summary_freq == 0:
            print("write summay...")
            # generate summary for tensorboard
            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, step)

        if step % args.save_pred_every == 0:
            print("save a predict as picture...")

            fig, axes = plt.subplots(args.save_num_images, 3, figsize=(16, 12))
            print("images type: {}".format(type(images)))
            print("labels type: {}".format(type(labels)))
            #print("preds_result_value type: {},shape {}".format(type(preds_result_value[0]),(preds_result_value[0]).get_shape()))

            print("preds_result shape: {}".format(preds_result_value.shape))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds_result_value[i, :, :, 0]))
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            save(writeSaver, sess, args.snapshot_dir, step)

        duration = time.time() - start_time
        print('step {:d} \t  ({:.3f} sec/step)'.format(step, duration))



    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
