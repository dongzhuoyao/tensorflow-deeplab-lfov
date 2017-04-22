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

BATCH_SIZE = 16
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
INPUT_SIZE = '321,321'
LEARNING_RATE = 1e-4
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000
RANDOM_SCALE = True
RESTORE_FROM = './deeplab_lfov.ckpt'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 20
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
    parser.add_argument("--recurrent_times", type=int, default=3,
                        help="recurrent_times"
                             "recurrent_times")

    parser.add_argument("--summary_freq", type=int, default=100,
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

    attention_map_placeholder = tf.placeholder(tf.float32, shape=[args.batch_size, h, w, 1])

    # Define the loss and optimisation parameters.
    main_loss_1, pre_upscaled_1, output_attention_map_1, main_loss_2, pre_upscaled_2,\
    output_attention_map_2, main_loss_3, pre_upscaled_3, output_attention_map_3  = net.loss(image_batch, label_batch,attention_map_placeholder)

    loss = main_loss_1+1*main_loss_2+1*main_loss_3

    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimiser.minimize(loss, var_list=trainable)

    tf.get_variable_scope().reuse_variables()
    pred_result = net.preds(image_batch,attention_map_placeholder)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        pre_upscaled_1_converted = convert(pre_upscaled_1)

    with tf.name_scope("convert_targets"):
        pre_upscaled_2_converted = convert(pre_upscaled_2)

    with tf.name_scope("convert_outputs"):
        pre_upscaled_3_converted = convert(pre_upscaled_3)


    #summary
    with tf.name_scope("loss_summary"):
        tf.summay.scalar("loss",loss)
        tf.summay.scalar("loss_1", main_loss_1)
        tf.summay.scalar("loss_2", main_loss_2)
        tf.summay.scalar("loss_3", main_loss_3)

    with tf.name_scope("image_summary"):
        tf.summary.image("origin", convert(image_batch))
        tf.summary.image("label", convert(label_batch))
        tf.summary.image("predict_1", pre_upscaled_1_converted)
        tf.summary.image("predict_2", pre_upscaled_2_converted)
        tf.summary.image("predict_3", pre_upscaled_3_converted)
        tf.summary.image('total',
                         tf.concat([convert(image_batch), convert(label_batch), pre_upscaled_1_converted,pre_upscaled_2_converted,pre_upscaled_3_converted], 2),
                         max_outputs=4)

    merged_summary_op = tf.summary.merge_all()

    sv = tf.train.Supervisor(logdir=args.summay_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:

        # Set up tf session and initialize variables.
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        # check the shape
        print("begin shape check....")
        for v in tf.global_variables():
            print("{}:  {}".format(v.name, v.get_shape()))

        # don't need initiate "filter_of_attention_map"!!!
        var_to_be_restored =  [x for x in trainable if u'filter_of_attention_map'.encode('utf-8') not in x.name.encode('utf-8')]
        # Saver for storing checkpoints of the model.
        for tmp in var_to_be_restored:
            print("variable name: {},type: {}".format(tmp.name,type(tmp.name)))

        saver = tf.train.Saver(var_list=var_to_be_restored, max_to_keep=40)
        if args.restore_from is not None:
           load(saver, sess, args.restore_from)


        #define Summary
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        print("parameter_count =", sess.run(parameter_count))

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Iterate over training steps.
        for step in range(1,args.num_steps):
            start_time = time.time()

            cur_imgs,cur_labels = sess.run([image_batch,label_batch])

            #do Loop recurrent training
            init_attention_map = np.zeros((args.batch_size,h,w,1), dtype=np.float32)

            _loss,_main_loss_1, _pre_upscaled_1, _output_attention_map_1, _main_loss_2, _pre_upscaled_2,\
    _output_attention_map_2, _main_loss_3, _pre_upscaled_3, _output_attention_map_3 = sess.run([loss,main_loss_1, pre_upscaled_1, output_attention_map_1, main_loss_2, pre_upscaled_2,\
    output_attention_map_2, main_loss_3, pre_upscaled_3, output_attention_map_3], feed_dict=
            {
             attention_map_placeholder: init_attention_map
             })


            print('step {:d} \t total_loss: {:.3f}, loss 1: {:.3f}, loss 2: {:.3f}, loss 3: {:.3f}'.format(step,_loss,_main_loss_1,_main_loss_2,_main_loss_3))

            if step % args.summary_freq == 0:
                print("write summay...")
                # generate summary for tensorboard
                summary_str = td.sess.run(merged_summary_op)
                sv.summary_computed(sess, summary_str)

            if step % args.save_pred_every == 0:
                images = cur_imgs
                labels = cur_labels
                #do predict
                preds_result_value = sess.run([pred_result],feed_dict=
                {
                 attention_map_placeholder:init_attention_map
                 })
                #single value
                preds_result_value =preds_result_value[0]

                fig, axes = plt.subplots(args.save_num_images, 3, figsize=(16, 12))
                print("images type: {}".format(type(images)))
                print("labels type: {}".format(type(labels)))
                #print("preds_result_value type: {},shape {}".format(type(preds_result_value[0]),(preds_result_value[0]).get_shape()))

                #print("preds_result shape: {}".format(preds_result_value.get_shape()))
                for i in xrange(args.save_num_images):
                    axes.flat[i * 3].set_title('data')
                    axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                    axes.flat[i * 3 + 1].set_title('mask')
                    axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                    axes.flat[i * 3 + 2].set_title('pred')
                    axes.flat[i * 3 + 2].imshow(decode_labels(preds_result_value[i, :, :, 0]))
                plt.savefig(args.save_dir + str(start_time) + ".png")
                plt.close(fig)
                save(saver, sess, args.snapshot_dir, step)

            duration = time.time() - start_time
            print('step {:d} \t  ({:.3f} sec/step)'.format(step, duration))



        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()
