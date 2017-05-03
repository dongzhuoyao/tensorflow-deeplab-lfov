from PIL import Image
import numpy as np

# Colour map.
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor



def single_channel_process(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    print("imgs.shape: {}".format(imgs.shape))
    n, h, w, c = imgs.shape#c=1,because attention map has only one channel
    #print ("single_channel_process imgs.shape: {}".format(imgs.shape))

    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        tmp = imgs[i].flatten()
        bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        tmp = np.digitize(tmp, bins)
        tmp = tmp*255.0/tmp.max()
        tmp = tmp.reshape((h,w)).astype(np.uint8)

        outputs[i, :, :, 0] = tmp
        outputs[i, :, :, 1] = tmp
        outputs[i, :, :, 2] = tmp
    return outputs


def decode_labels(mask,real=False,show_confusion=False):
    """Decode batch of segmentation masks.
    
    Args:
      label_batch: result of inference after taking argmax.
    
    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < 21:
                if real:
                    pixels[k_, j_] = (k,k,k)
                else:
                    pixels[k_,j_] = label_colours[k]
            else:
                if show_confusion:
                    pixels[k_, j_]=(255,255,255)
                else:
                    pass
    return np.array(img)


