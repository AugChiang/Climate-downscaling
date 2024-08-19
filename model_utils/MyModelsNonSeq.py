from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, UpSampling2D, Activation, BatchNormalization, Input
from tensorflow.keras.layers import MaxPool2D, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import math
from . import MyLayersNonSeq

def MyModelV2(config:object,
              extract_every_n_layer=2,
              aux=None,
              use_elevation = True,
            ):

    x = Input(shape=(config.input_height,
                     config.input_width,
                     config.num_channel)) # inputs
    x0 = x
    x2 = aux # elevation

    if(config.skip_connection):
        x3 = UpSampling2D(size=(config.scale, config.scale),
                          interpolation='bilinear')(x0)
    else:
        x3 = None

    # conv_feature = []
    attention_map = []
    for i in range(config.num_main_layers):
        x = Conv2D(filters = 64,
                   kernel_size = (3,3),
                   strides = 1,
                   padding='SAME', activation='relu')(x)
        if(i % extract_every_n_layer == 0):
            # Attention Block work on main-stream line of x
            x_attention = tf.math.multiply(x, MyLayersNonSeq.ChAttenBlock(x))
            x = x + tf.math.multiply(x_attention, MyLayersNonSeq.SpAttenBlock(x_attention))
            x_attention = MyLayersNonSeq.CreateConv(input=x,
                                                    filters = 1,
                                                    size=(3,3),
                                                    apply_batchnorm=config.batch_norm) # reduce # of parameters
            attention_map.append(x_attention)

    # Fusion of feature maps  
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = MyLayersNonSeq.CreateAuxConcat(input=x,
                                       output_ch=config.num_channel,
                                       apply_batchnorm=config.batch_norm)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(config.upsample == 'bilinear'):
        x = MyLayersNonSeq.CreateUpsample(input=x,
                                          scale=config.scale,
                                          interpolation='bilinear')
    elif(config.upsample == 'bicubic'):
        x = MyLayersNonSeq.CreateUpsample(input=x,
                                          scale=config.scale,
                                          interpolation='bicubic')
    elif(config.upsample == 'subpixel'):
        x = MyLayersNonSeq.CreateSubPixelUpsample(input=x,
                                                  size=(5,5),
                                                  scale=config.scale,
                                                  apply_batchnorm=config.batch_norm)
    else:
        x = MyLayersNonSeq.CreateDeconvUpsample(input=x,
                                                size=(5,5),
                                                scale=config.scale,
                                                apply_batchnorm=config.batch_norm)
    
    x = MyLayersNonSeq.CreateConv(input=x,
                                  filters=64,
                                  size=(3,3),
                                  apply_batchnorm=config.batch_norm)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])

    if(use_elevation and (x2 is not None)):
        '''https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat'''
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])

    x = MyLayersNonSeq.CreateAuxConcat(input=x,
                                       output_ch=1,
                                       apply_batchnorm=config.batch_norm)

    return tf.keras.Model(inputs=x0, outputs=x)