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
    
    
    def _create_network(self, input_batch,keep_prob):
        """Construct DeepLab-LargeFOV network.
        
        Args:
          input_batch: batch of pre-processed images.
          keep_prob: probability of keeping neurons intact.
          
        Returns:
          A downsampled segmentation mask. 
        """
        current = input_batch
        
        v_idx = 0 # Index variable.
        is_deal_first_layer = 0
        
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
                v_idx += 1
            # Optional pooling and dropout after each block.
            if b_idx < 3:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current, 
                             ksize=[1, ks, ks, 1],
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                current = tf.nn.avg_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        
        # Classification layer; no ReLU.
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current



    
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

    def _create_reusable_nework(self,img_batch):
        with tf.variable_scope('resusable_network', reuse=True):
            result = self._create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        return result

      
    def preds(self, input_batch):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        print("predicts: input_batch_shape: {}".format(input_batch.get_shape()))
        raw_output = self._create_network(tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3,])
        raw_output = tf.argmax(raw_output, dimension=3)
        raw_output = tf.expand_dims(raw_output, dim=3) # Create 4D-tensor.
        print("predicts: raw_output: {}".format(raw_output.get_shape()))
        return tf.cast(raw_output, tf.uint8)

    def RAU(self, img_batch, label_batch,pre_attention_map):

        #flatten it in order to do softmax
        pre_attention_map = tf.reshape(pre_attention_map,[1,-1])
        pre_attention_map = tf.nn.softmax(pre_attention_map)
        #add exponential operation
        pre_attention_map = tf.exp(pre_attention_map)
        #restore to original shape
        pre_attention_map = tf.reshape(pre_attention_map,label_batch.get_shape()[0:3])
        print("pre_attention_map shape: {}".format(pre_attention_map.get_shape()))
        pre_attention_map = tf.expand_dims(pre_attention_map,axis=3)
        print("pre_attention_map afater expand_dims shape: {}".format(pre_attention_map.get_shape()))
        pre_attention_map = tf.concat([pre_attention_map,pre_attention_map,pre_attention_map],axis=-1)
        print("pre_attention_map afater concat shape: {}".format(pre_attention_map.get_shape()))
        #apply attention map
        img_batch = tf.multiply(img_batch,pre_attention_map)


        #raw_output = self._create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        raw_output = self._create_reusable_nework(img_batch)

        pre_upscaled_4d = predict_4d = tf.image.resize_bilinear(raw_output, tf.shape(img_batch)[1:3, ])
        pre_upscaled_4d = tf.argmax(pre_upscaled_4d, dimension=3)
        pre_upscaled_4d = tf.expand_dims(pre_upscaled_4d, dim=3)  # from 3-D to 4-D
        pre_upscaled_4d = tf.cast(pre_upscaled_4d,tf.uint8)

        print "predict before reshape: {}".format(raw_output.get_shape())
        # turn a matrix into a vector!!!
        prediction = tf.reshape(raw_output, [-1, n_classes])
        print "predict shape: {}".format(prediction.get_shape())

        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])
        print "gt shape: {}".format(gt.get_shape())

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        main_loss = tf.reduce_mean(loss)

        gt_upscaled = tf.image.resize_bilinear(label_batch, tf.shape(img_batch)[1:3, ])
        gt_upscaled = tf.argmax(gt_upscaled, dimension=3)
        gt_upscaled = tf.expand_dims(gt_upscaled, dim=3)  # from 3-D to 4-D
        gt_upscaled = tf.cast(gt_upscaled, tf.uint8)

        #calculate attention map for next recurrent use
        predict_3d =tf.reduce_max(predict_4d,keep_dims=True,axis=3)
        predict_3d_inverse = tf.subtract(tf.constant(1.0),predict_3d)

        att_3d = tf.cast(tf.not_equal(gt_upscaled, pre_upscaled_4d), tf.float32)
        att_3d_inverse = tf.cast(tf.equal(gt_upscaled, pre_upscaled_4d), tf.float32)


        output_attention_map = tf.add(tf.multiply(predict_3d,att_3d),tf.multiply(predict_3d_inverse,att_3d_inverse))

        print "attention_map size: {}".format(output_attention_map.get_shape())

        #pre_upscaled_unit8 = tf.cast(pre_upscaled,tf.uint8)
        return main_loss,pre_upscaled_4d,output_attention_map,predict_3d

    
    def loss(self, img_batch, label_batch):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """
        #init attention map
        init_attention_map = tf.zeros(img_batch.get_shape()[0:3], tf.float32)
        print("init_attention_map shape: {}".format(init_attention_map.get_shape()))
        main_loss_1, pre_upscaled_1, output_attention_map_1,predict_3d_1 = self.RAU(img_batch, label_batch,init_attention_map)
        main_loss_2, pre_upscaled_2, output_attention_map_2,predict_3d_2 = self.RAU(img_batch, label_batch, output_attention_map_1)
        main_loss_3, pre_upscaled_3, output_attention_map_3,predict_3d_3= self.RAU(img_batch, label_batch, output_attention_map_2)


        return main_loss_1, pre_upscaled_1, output_attention_map_1,predict_3d_1,main_loss_2, pre_upscaled_2, \
               output_attention_map_2,predict_3d_2,main_loss_3, pre_upscaled_3, output_attention_map_3,predict_3d_3
