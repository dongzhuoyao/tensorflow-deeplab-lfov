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

    def stage_network(self,stage_name,current):

        w_initialiser = tf.random_normal_initializer(mean=0,stddev=0.01)
        b_initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
        #533 = 512+21
        stage1_c1_w = tf.Variable(w_initialiser(shape=(7, 7, 533, 128)), name=stage_name+"_c1_w")
        stage1_c1_b = tf.Variable(b_initialiser(shape=(128,)), name=stage_name+"_c1_b")
        current = tf.nn.conv2d(current, stage1_c1_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.relu(tf.nn.bias_add(current, stage1_c1_b))

        stage1_c2_w = tf.Variable(w_initialiser(shape=(7, 7, 128, 128)), name=stage_name+"_c2_w")
        stage1_c2_b = tf.Variable(b_initialiser(shape=(128,)), name=stage_name+"_c2_b")
        current = tf.nn.conv2d(current, stage1_c2_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.relu(tf.nn.bias_add(current, stage1_c2_b))

        stage1_c3_w = tf.Variable(w_initialiser(shape=(7, 7, 128, 128)), name=stage_name+"_c3_w")
        stage1_c3_b = tf.Variable(b_initialiser(shape=(128,)), name=stage_name+"_c3_b")
        current = tf.nn.conv2d(current, stage1_c3_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.relu(tf.nn.bias_add(current, stage1_c3_b))

        stage1_c4_w = tf.Variable(w_initialiser(shape=(7, 7, 128, 128)), name=stage_name+"_c4_w")
        stage1_c4_b = tf.Variable(b_initialiser(shape=(128,)), name=stage_name+"_c4_b")
        current = tf.nn.conv2d(current, stage1_c4_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.relu(tf.nn.bias_add(current, stage1_c4_b))

        stage1_c5_w = tf.Variable(w_initialiser(shape=(7, 7, 128, 128)), name=stage_name+"_c5_w")
        stage1_c5_b = tf.Variable(b_initialiser(shape=(128,)), name=stage_name+"_c5_b")
        current = tf.nn.conv2d(current, stage1_c5_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.relu(tf.nn.bias_add(current, stage1_c5_b))

        stage1_c6_w = tf.Variable(w_initialiser(shape=(1, 1, 128, 128)), name=stage_name+"_c6_w")
        stage1_c6_b = tf.Variable(b_initialiser(shape=(128,)), name=stage_name+"_c6_b")
        current = tf.nn.conv2d(current, stage1_c6_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.relu(tf.nn.bias_add(current, stage1_c6_b))

        stage1_c7_w = tf.Variable(w_initialiser(shape=(1, 1, 128, 21)), name=stage_name+"_c7_w")
        stage1_c7_b = tf.Variable(b_initialiser(shape=(21,)), name=stage_name+"_c7_b")
        current = tf.nn.conv2d(current, stage1_c7_w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(current, stage1_c7_b)

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
        
        v_idx = 0 # Index variable.
        conv4_3 = None
        
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
            if b_idx ==3:
                conv4_3 = current
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
        
        # Classification layer; no ReLU. #
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = stage0_feat = tf.nn.bias_add(conv, b)

        #stage 1
        current = tf.concat([current,conv4_3],axis=-1)
        current = stage1_feat =  self.stage_network("stage1",current)

        #stage 2
        current = tf.concat([current, conv4_3], axis=-1)
        current = stage2_feat = self.stage_network("stage2", current)

        #stage 3
        current = tf.concat([current, conv4_3], axis=-1)
        current = stage3_feat = self.stage_network("stage3", current)

        return stage0_feat,stage1_feat,stage2_feat,stage3_feat
    
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
      
    def preds(self, input_batch):
        """Create the network and run inference on the input batch.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        _,_,_,raw_output = self._create_network(tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))

        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3,])
        raw_output = tf.argmax(raw_output, dimension=3)
        raw_output = tf.expand_dims(raw_output, dim=3) # Create 4D-tensor.
        return tf.cast(raw_output, tf.uint8)
        
    def cal_loss(self,raw_output,label_batch):
        prediction = tf.reshape(raw_output, [-1, n_classes])
        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, n_classes])
        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)
        return reduced_loss

    def loss(self, img_batch, label_batch,weight_decay = 0.05):
        """Create the network, run inference on the input batch and compute loss.
        
        Args:
          input_batch: batch of pre-processed images.
          
        Returns:
          Pixel-wise softmax loss.
        """
        raw_output,raw_output1,raw_output2,raw_output3 = self._create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        loss0 = self.cal_loss(raw_output,label_batch)
        loss1 = self.cal_loss(raw_output1, label_batch)
        loss2 = self.cal_loss(raw_output2, label_batch)
        loss3 = self.cal_loss(raw_output3,label_batch)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = loss0+loss1+loss2+loss3 + weight_decay * sum(reg_losses)

        return loss
