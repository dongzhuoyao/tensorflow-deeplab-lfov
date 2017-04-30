import tensorflow as tf
from six.moves import cPickle

# Loading net skeleton with parameters name and shapes.
with open("./util/net_skeleton.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)

# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool](stride=1)
##       -> [conv-relu](dilation=12, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=21) -> [pixel-wise softmax loss].
num_layers    = [2, 2, 3, 3, 3, 1, 1, 1]
dilations     = [[1, 1],
                 [1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [2, 2, 2],
                 [12], 
                 [1], 
                 [1]]
n_classes = 21
# All convolutional and pooling operations are applied using kernels of size 3x3; 
# padding is added so that the output of the same size as the input.
ks = 3

def create_variable(name, shape):
    """Create a convolution filter variable of the given name and shape,
       and initialise it using Xavier initialisation 
       (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    """Create a bias variable of the given name and shape,
       and initialise it to zero.
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

class DeepLabLFOVModel(object):
    """DeepLab-LargeFOV model with atrous convolution and bilinear upsampling.
    
    This class implements a multi-layer convolutional neural network for semantic image segmentation task.
    This is the same as the model described in this paper: https://arxiv.org/abs/1412.7062 - please look
    there for details.
    """
    
    def __init__(self, weights_path=None):
        """Create the model.
        
        Args:
          weights_path: the path to the cpkt file with dictionary of weights from .caffemodel.
        """
        self.variables = self._create_variables(weights_path)
        
    def _create_variables(self, weights_path):
        """Create all variables used by the network.
        This allows to share them between multiple calls 
        to the loss function.
        
        Args:
          weights_path: the path to the ckpt file with dictionary of weights from .caffemodel. 
                        If none, initialise all variables randomly.
        
        Returns:
          A dictionary with all variables.
        """
        var = list()
        index = 0
        
        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f) # Load pre-trained weights.
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],
                                           name=name))
                del weights
        else:
            # Initialise all weights randomly with the Xavier scheme,
            # and 
            # all biases to 0's.
            for name, shape in net_skeleton:
                if "/w" in name: # Weight filter.
                    w = create_variable(name, list(shape))
                    var.append(w)
                else:
                    b = create_bias_variable(name, list(shape))
                    var.append(b)
        return var

    def upsample_double(self, name, current,output_shape):
        upsample_factor = 2
        #x = tf.pad(current, [[0, 0], [upsample_factor - 1, upsample_factor - 1], [upsample_factor - 1, upsample_factor - 1], [0, 0]], mode='SYMMETRIC')
        #out_shape = tf.shape(x) * tf.constant([1, upsample_factor, upsample_factor, 1], tf.int32)

        #filter_shape = 2 * upsample_factor

        w = tf.get_variable(name=name, shape=[3, 3, 1, 1],
                            initializer=tf.contrib.layers.xavier_initializer())
        current = tf.nn.conv2d_transpose(current, w,
                                        output_shape=output_shape,
                                        strides=[1, upsample_factor, upsample_factor, 1], padding="SAME")
        return current
    
    def _create_network(self, input_batch, keep_prob):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.
          
        Returns:
          A downsampled segmentation mask. 
        """
        current = input_batch


        feature_map_list_for_debug = []
        
        v_idx = 0 # Index variable.
        
        # Last block is the classification layer.
        for b_idx in xrange(len(dilations) - 1):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                feature_map_list_for_debug.append(current)
                v_idx += 1

            if b_idx == 0:
                w = tf.get_variable(name="block1/w", shape=[3,3,64,1],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('block1/b', [1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                block1 = tf.nn.conv2d(current,w,strides=[1,1,1,1],padding='SAME')
                block1 = tf.nn.bias_add(block1, b)
                #houmian tongyi jia sigmoid
            elif b_idx ==1:
                w = tf.get_variable(name="block2/w", shape=[3, 3, 128, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('block2/b', [1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                block2 = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                block2 = tf.nn.bias_add(block2, b)

                block2 = self.upsample_double("block2/deconv1/w", block2, [tf.shape(input_batch)[0],321,321,1])
                # houmian tongyi jia sigmoid
            elif b_idx ==2:
                w = tf.get_variable(name="block3/w", shape=[3, 3, 256, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('block3/b', [1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                block3 = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                block3 = tf.nn.bias_add(block3, b)


                block3 = self.upsample_double("block3/deconv1/w", block3, [tf.shape(input_batch)[0],161,161,1])
                block3 = self.upsample_double("block3/deconv2/w", block3, [tf.shape(input_batch)[0],321,321,1])
                # houmian tongyi jia sigmoid
            elif b_idx ==3:
                w = tf.get_variable(name="block4/w", shape=[3, 3, 512, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('block4/b', [1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                block4 = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                block4 = tf.nn.bias_add(block4, b)

                block4 = self.upsample_double("block4/deconv1/w", block4, [tf.shape(input_batch)[0],81,81,1])
                block4 = self.upsample_double("block4/deconv2/w", block4, [tf.shape(input_batch)[0],161,161,1])
                block4 = self.upsample_double("block4/deconv3/w", block4, [tf.shape(input_batch)[0],321,321,1])
                # houmian tongyi jia sigmoid
            elif b_idx ==4:
                w = tf.get_variable(name="block5/w", shape=[3, 3, 512, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('block5/b', [1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                block5 = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                block5 = tf.nn.bias_add(block5, b)

                block5 = self.upsample_double("block5/deconv1/w", block5, [tf.shape(input_batch)[0],81,81,1])
                block5 = self.upsample_double("block5/deconv2/w", block5, [tf.shape(input_batch)[0],161,161,1])
                block5 = self.upsample_double("block5/deconv3/w", block5, [tf.shape(input_batch)[0],321,321,1])
                # houmian tongyi jia sigmoid
            else:
                pass


            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
                feature_map_list_for_debug.append(current)
            elif b_idx == 3:
                current = tf.nn.max_pool(current, 
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
                feature_map_list_for_debug.append(current)
            elif b_idx == 4:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                feature_map_list_for_debug.append(current)
                current = tf.nn.avg_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                feature_map_list_for_debug.append(current)
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
                feature_map_list_for_debug.append(current)
        
        # Classification layer; no ReLU.
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        w_final_map = tf.get_variable(name="block_main/w", shape=[3, 3, 5, 1],
                            initializer=tf.contrib.layers.xavier_initializer())
        b_final_map = tf.get_variable('block_main/b', [1], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        final_map = tf.nn.conv2d(tf.concat([block1, block2, block3, block4, block5], 3),w_final_map,strides=[1, 1, 1, 1], padding='SAME')
        final_map = tf.nn.bias_add(final_map, b_final_map)

        print("====feature map check====")
        for v in feature_map_list_for_debug:
            print("{}:  {}".format(v.name, v.get_shape()))


        return current,[block1, block2, block3, block4, block5,final_map]
    
    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.

        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.

        Returns:
          Outputs a tensor of shape [batch_size h w 21]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # As labels are integer numbers, need to use NN interp.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # Reducing the channel dimension.
            input_batch = tf.one_hot(input_batch, depth=21)
        return input_batch

    
    def loss(self, img_batch, label_batch,weight_decay = 0.05):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """
        raw_output,hed_predict_list = self._create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        prediction = tf.reshape(raw_output, [-1, n_classes])

        org_label_batch = label_batch
        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])
        
        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = reduced_loss

        #calculate confusion attention map

        gt_4d_label = org_label_batch
        gt_4d_label = tf.cast(gt_4d_label, tf.uint8)

        predict_4d = tf.image.resize_bilinear(raw_output, tf.shape(img_batch)[1:3, ])
        predict_4d = tf.nn.softmax(predict_4d) #convert to number between 0-1
        predict_4d_label = tf.expand_dims(tf.argmax(predict_4d, dimension=3),dim=3)
        predict_4d_label = tf.cast(predict_4d_label, tf.uint8)

        predict_4d_pro = confidence_map =  tf.reduce_max(predict_4d, keep_dims=True, axis=3)
        predict_4d_pro_inverse = tf.subtract(tf.constant(1.0), predict_4d_pro)

        att_4d = tf.cast(tf.not_equal(gt_4d_label, predict_4d_label), tf.float32)
        att_4d_inverse = tf.cast(tf.equal(gt_4d_label, predict_4d_label), tf.float32)

        attention_map_gt = tf.add(tf.multiply(predict_4d_pro, att_4d), tf.multiply(predict_4d_pro_inverse, att_4d_inverse))


        #confusion attention map loss
        costs = []
        for idx, b in enumerate(hed_predict_list):
            b = tf.nn.sigmoid(b, name='hed-output{}'.format(idx + 1))
            bcost = tf.reduce_mean(tf.square(b - attention_map_gt),name="hed-loss-{}".format(idx+1))
            costs.append(bcost)
        hed_total_cost = tf.add_n(costs, name='hed-total-loss')

        return reduced_loss,hed_total_cost,b,attention_map_gt,confidence_map
