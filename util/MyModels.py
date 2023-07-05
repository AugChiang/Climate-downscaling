from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, UpSampling2D, Activation, BatchNormalization, Input
from tensorflow.keras.layers import MaxPool2D, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import math
import MyLayers
#===============================================================================================
#========================================= MyModel =============================================
#===============================================================================================
def MyModelV2(n_layers, scale, xn, xm, ch, extract_every_n_layer=2, Upsample='subpixel',
              aux=None, use_elevation = True, use_skip_connect=True, BatchNorm=False):
    conv_layers = []
    conv_layers.append(Sequential([
                        Conv2D(filters = 64,
                               kernel_size = (3,3),
                               strides = 1,
                               padding='SAME'),
                        Activation('relu')]))

    for i in range(n_layers-1):
        layer = MyLayers.CreateConv(apply_batchnorm=BatchNorm)
        conv_layers.append(layer)

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation

    if(use_skip_connect):
        x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)
    else:
        x3 = None

    # conv_feature = []
    attention_map = []
    for i in range(n_layers):
        x = conv_layers[i](x)
        if(i % extract_every_n_layer == 0):
            # Attention Block work on main-stream line of x
            x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
            x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))
            conv = MyLayers.CreateConv(filters = 1, size=(3,3), apply_batchnorm=BatchNorm) # reduce # of parameters
            x_attention = conv(x)
            attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat(output_ch=ch, apply_batchnorm=BatchNorm)
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(Upsample == 'bilinear'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bilinear')
    elif(Upsample == 'bicubic'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bicubic')
    elif(Upsample == 'subpixel'):
        upsampling_layer = MyLayers.CreateSubPixelUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)
    else:
        upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)

    conv2 = MyLayers.CreateConv(filters = 64, size=(3,3), apply_batchnorm=BatchNorm)
    aux_layer = MyLayers.CreateAuxConcat(output_ch=1, apply_batchnorm=BatchNorm)
    x = upsampling_layer(x)
    x = conv2(x)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])
  
    if(use_elevation and (x2 is not None)):
        '''https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat'''
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])

    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
# =====================================================================================
def MyRDB(n_rdb_block, n_rdb_layers, scale, xn, xm, ch, Upsample='subpixel',
          aux=None, use_elevation = True, BatchNorm=False):

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation
    x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)

    attention_map = []
    x = Conv2D(filters=64, kernel_size=(3,3),
               strides= 1, padding='SAME', activation='relu')(x)
    # # Attention Block work on main-stream line of x
    # x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
    # x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))
    # # reduce paramter
    x_attention = Conv2D(filters=1, kernel_size=(3,3),
                         strides=1, padding='SAME', activation='relu')(x)
    attention_map.append(x_attention)

    # ResBlock
    for i in range(n_rdb_block):
        x = MyLayers.ResidualDenseBlock(input_x=x, n_layers=n_rdb_layers,
                               filters=64, size=(3,3), strides=1)
        
        # # Attention Block work on main-stream line of x
        # x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
        # x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))

        # # reduce paramter and extract
        x_attention = Conv2D(filters=1, kernel_size=(3,3),
                             strides=1, padding='SAME', activation='relu')(x)
        attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat(output_ch=ch)
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(Upsample == 'bilinear'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bilinear')
    elif(Upsample == 'bicubic'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bicubic')
    elif(Upsample == 'subpixel'):
        upsampling_layer = MyLayers.CreateSubPixelUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)
    else:
        upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)

    x = upsampling_layer(x)
    x = Conv2D(filters=64, kernel_size=(3,3),
               strides= 1, padding='SAME', activation='relu')(x)

    # concat with interpolation
    x = Concatenate(axis=-1)([x, x3])

    # concat with elevation
    repeat_shape = tf.shape(x)
    topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
    x = Concatenate(axis=-1)([x, topo_batch])

    # fushion
    aux_layer = MyLayers.CreateAuxConcat(output_ch=1)
    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
# =====================================================================================
def MyRDB_RAB(n_rdb_block, n_rdb_layers, scale, xn, xm, ch, Upsample='subpixel',
          aux=None, use_elevation = True, BatchNorm=False):

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation
    x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)

    attention_map = []
    x = Conv2D(filters=64, kernel_size=(3,3),
               strides= 1, padding='SAME', activation='relu')(x)
    # # Attention Block work on main-stream line of x
    # x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
    # x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))
    # # reduce paramter
    x_attention = Conv2D(filters=1, kernel_size=(3,3),
                         strides=1, padding='SAME', activation='relu')(x)
    attention_map.append(x_attention)

    # ResBlock
    for i in range(n_rdb_block):
        x = MyLayers.ResidualDenseBlock(input_x=x, n_layers=n_rdb_layers,
                               filters=64, size=(3,3), strides=1)
        
        # Attention Block work on main-stream line of x
        x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
        x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))

        # reduce paramter and extract
        x_attention = Conv2D(filters=1, kernel_size=(3,3),
                             strides=1, padding='SAME', activation='relu')(x)
        attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat(output_ch=ch)
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(Upsample == 'bilinear'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bilinear')
    elif(Upsample == 'bicubic'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bicubic')
    elif(Upsample == 'subpixel'):
        upsampling_layer = MyLayers.CreateSubPixelUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)
    else:
        upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)

    x = upsampling_layer(x)
    x = Conv2D(filters=64, kernel_size=(3,3),
               strides= 1, padding='SAME', activation='relu')(x)

    # concat with interpolation
    x = Concatenate(axis=-1)([x, x3])

    # concat with elevation
    repeat_shape = tf.shape(x)
    topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
    x = Concatenate(axis=-1)([x, topo_batch])

    # fushion
    aux_layer = MyLayers.CreateAuxConcat(output_ch=1)
    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
# =====================================================================================
def NoConvAfterUpscale(n_layers, scale, xn, xm, ch, aux=None, use_elevation = True, use_skip_connect=True):
    conv_layers = []
    conv_layers.append(Sequential([
                        Conv2D(filters = 64,
                               kernel_size = (3,3),
                               strides = 1,
                               padding='SAME'),
                        Activation('relu')]))

    for i in range(n_layers-1):
        layer = MyLayers.CreateConv()
        conv_layers.append(layer)

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation

    if(use_skip_connect):
        x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)
    else:
        x3 = None

    # conv_feature = []
    attention_map = []
    for i in range(n_layers):
        x = conv_layers[i](x)
        if(i % 2 == 0):
            # Attention Block work on main-stream line of x
            x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
            x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))
            conv = MyLayers.CreateConv(filters = 1, size=(3,3)) # reduce # of parameters
            x_attention = conv(x)
            attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat()
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # upsampling_layer = MyLayers.CreateUpsample(scale=scale)
    upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale)
    # conv2 = MyLayers.CreateConv(filters = 64, size=(3,3))
    aux_layer = MyLayers.CreateAuxConcat()
    x = upsampling_layer(x)
    # x = conv2(x)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])
    
    if(use_elevation and (x2 is not None)):
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])

    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
# =====================================================================================
def NoAtten(n_layers, scale, xn, xm, ch, extract_every_n_layer=2, Upsample='subpixel',
              aux=None, use_elevation = True, use_skip_connect=True, BatchNorm=False):
    conv_layers = []
    conv_layers.append(Sequential([
                        Conv2D(filters = 64,
                               kernel_size = (3,3),
                               strides = 1,
                               padding='SAME'),
                        Activation('relu')]))

    for i in range(n_layers-1):
        layer = MyLayers.CreateConv(apply_batchnorm=BatchNorm)
        conv_layers.append(layer)

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation

    if(use_skip_connect):
        x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)
    else:
        x3 = None

    # conv_feature = []
    attention_map = []
    for i in range(n_layers):
        x = conv_layers[i](x)
        if(i % extract_every_n_layer == 0):
            # Attention Block work on main-stream line of x
            # x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
            # x = x + tf.math.multiply(x_attention, MyLayers.SpAttenBlock(x_attention))
            conv = MyLayers.CreateConv(filters = 1, size=(3,3), apply_batchnorm=BatchNorm) # reduce # of parameters
            x_attention = conv(x)
            attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat(output_ch=ch, apply_batchnorm=BatchNorm)
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(Upsample == 'bilinear'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bilinear')
    elif(Upsample == 'bicubic'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bicubic')
    elif(Upsample == 'subpixel'):
        upsampling_layer = MyLayers.CreateSubPixelUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)
    else:
        upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)

    conv2 = MyLayers.CreateConv(filters = 64, size=(3,3), apply_batchnorm=BatchNorm)
    aux_layer = MyLayers.CreateAuxConcat(output_ch=1, apply_batchnorm=BatchNorm)
    x = upsampling_layer(x)
    x = conv2(x)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])
  
    if(use_elevation and (x2 is not None)):
        '''https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat'''
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])

    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
# =====================================================================================
def NoSAB(n_layers, scale, xn, xm, ch, extract_every_n_layer=2, Upsample='subpixel',
              aux=None, use_elevation = True, use_skip_connect=True, BatchNorm=False):
    conv_layers = []
    conv_layers.append(Sequential([
                        Conv2D(filters = 64,
                               kernel_size = (3,3),
                               strides = 1,
                               padding='SAME'),
                        Activation('relu')]))

    for i in range(n_layers-1):
        layer = MyLayers.CreateConv(apply_batchnorm=BatchNorm)
        conv_layers.append(layer)

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation

    if(use_skip_connect):
        x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)
    else:
        x3 = None

    # conv_feature = []
    attention_map = []
    for i in range(n_layers):
        x = conv_layers[i](x)
        if(i % extract_every_n_layer == 0):
            # Attention Block work on main-stream line of x
            x = x + tf.math.multiply(x, MyLayers.ChAttenBlock(x))
            conv = MyLayers.CreateConv(filters = 1, size=(3,3), apply_batchnorm=BatchNorm) # reduce # of parameters
            x_attention = conv(x)
            attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat(output_ch=ch, apply_batchnorm=BatchNorm)
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(Upsample == 'bilinear'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bilinear')
    elif(Upsample == 'bicubic'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bicubic')
    elif(Upsample == 'subpixel'):
        upsampling_layer = MyLayers.CreateSubPixelUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)
    else:
        upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)

    conv2 = MyLayers.CreateConv(filters = 64, size=(3,3), apply_batchnorm=BatchNorm)
    aux_layer = MyLayers.CreateAuxConcat(output_ch=1, apply_batchnorm=BatchNorm)
    x = upsampling_layer(x)
    x = conv2(x)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])
  
    if(use_elevation and (x2 is not None)):
        '''https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat'''
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])

    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
# =====================================================================================
def NoCAB(n_layers, scale, xn, xm, ch, extract_every_n_layer=2, Upsample='subpixel',
              aux=None, use_elevation = True, use_skip_connect=True, BatchNorm=False):
    conv_layers = []
    conv_layers.append(Sequential([
                        Conv2D(filters = 64,
                               kernel_size = (3,3),
                               strides = 1,
                               padding='SAME'),
                        Activation('relu')]))

    for i in range(n_layers-1):
        layer = MyLayers.CreateConv(apply_batchnorm=BatchNorm)
        conv_layers.append(layer)

    x = Input(shape=(xn, xm, ch)) # inputs
    x0 = x
    x2 = aux # elevation

    if(use_skip_connect):
        x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x0)
    else:
        x3 = None

    # conv_feature = []
    attention_map = []
    for i in range(n_layers):
        x = conv_layers[i](x)
        if(i % extract_every_n_layer == 0):
            # Attention Block work on main-stream line of x
            # x_attention = tf.math.multiply(x, MyLayers.ChAttenBlock(x))
            x = x + tf.math.multiply(x, MyLayers.SpAttenBlock(x))
            conv = MyLayers.CreateConv(filters = 1, size=(3,3), apply_batchnorm=BatchNorm) # reduce # of parameters
            x_attention = conv(x)
            attention_map.append(x_attention)

    # Fusion of feature maps
    fusion_map_layer = MyLayers.CreateAuxConcat(output_ch=ch, apply_batchnorm=BatchNorm)
    xa = Concatenate(axis=-1)([xa for xa in attention_map])
    x = Concatenate(axis=-1)([x, xa])
    x = fusion_map_layer(x)
    x = x + x0
    x = Activation('relu')(x)

    # Upsampling Layer (Upsampling methods)
    if(Upsample == 'bilinear'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bilinear')
    elif(Upsample == 'bicubic'):
        upsampling_layer = MyLayers.CreateUpsample(scale=scale, interpolation='bicubic')
    elif(Upsample == 'subpixel'):
        upsampling_layer = MyLayers.CreateSubPixelUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)
    else:
        upsampling_layer = MyLayers.CreateDeconvUpsample(size=(5,5), scale=scale, apply_batchnorm=BatchNorm)

    conv2 = MyLayers.CreateConv(filters = 64, size=(3,3), apply_batchnorm=BatchNorm)
    aux_layer = MyLayers.CreateAuxConcat(output_ch=1, apply_batchnorm=BatchNorm)
    x = upsampling_layer(x)
    x = conv2(x)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])
  
    if(use_elevation and (x2 is not None)):
        '''https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat'''
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])

    x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)
#===============================================================================================
#========================================= YNet ================================================
#===============================================================================================
def YNet(n_layers, scale, xn, xm, ch, aux=None, use_elevation = True):
    conv_layers= [] 
    deconv_layers = []
    conv_layers.append(Sequential([
                            Conv2D(filters = 64,
                                   kernel_size = (3,3),
                                   strides = 1,
                                   padding='SAME'),
                            Activation('relu')])
                        )

    for i in range(n_layers-1):
        layer = MyLayers.CreateConv(filters=64, size=(3,3))
        conv_layers.append(layer)
    # print("Conv Layers Len: ",len(conv_layers))
    for i in range(n_layers-1):
        layer = MyLayers.CreateDeconv(filters=64, output_ch=64, size=(3,3))
        deconv_layers.append(layer)
    
    deconv_layers.append(Sequential([
                            Conv2DTranspose(filters = 64,
                                            kernel_size = (3,3),
                                            strides = 1,
                                            padding='SAME'),
                            Activation('relu')]))

    # print("Deconv Layers Len: ",len(deconv_layers))
    x = Input(shape=[xn, xm, ch]) # inputs
    x0 = x
    x2 = aux # elevation
    x3 = UpSampling2D(size=(scale, scale), interpolation='bilinear')(x)

    conv_feature = []
    for i in range(n_layers):
        x = conv_layers[i](x)
        if((i+1) % 2 == 0 and len(conv_feature))<math.ceil(n_layers/2)-1:
            conv_feature.append(x)
    
    conv_ix = 0
    for i in range(n_layers):
        x = deconv_layers[i](x)
        if(i+1+n_layers) % 2 ==0 and conv_ix < len(conv_feature):
            conv_f = conv_feature[-(conv_ix + 1)]
            conv_ix += 1
            x = x + conv_f
            x = Activation('relu')(x)
    x = Conv2D(filters = ch, kernel_size = (3,3), strides = 1, padding='SAME')(x)
    x = x + x0
    x = Activation('relu')(x)
    
    upsampling_layer = MyLayers.CreateUpsample(filters=ch, scale=scale, interpolation='bilinear')
    x = upsampling_layer(x)

    if(x3 is not None):
        x = Concatenate(axis=-1)([x, x3])
    
    if(use_elevation and (x2 is not None)):
        aux_layer = MyLayers.CreateAuxConcat(output_ch=1)
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(x2, repeat_shape[0], axis=0)
        x = Concatenate(axis=-1)([x, topo_batch])
        x = aux_layer(x)

    return tf.keras.Model(inputs=x0, outputs=x)

#===============================================================================================
#========================================= SRCNN ===============================================
#===============================================================================================
def SRCNN(xn, xm, ch, scale, aux):

    conv1 = MyLayers.CreateConv(filters=64, size=(9,9), apply_batchnorm=False)
    conv2 = MyLayers.CreateConv(filters=32, size=(1,1), apply_batchnorm=False)

    x0 = tf.keras.Input(shape=(xn, xm, ch))
    repeat_shape = tf.shape(x0)
    topo_batch = tf.repeat(aux, repeat_shape[0], axis=0)
    x = UpSampling2D(size=(scale, scale), interpolation='bicubic')(x0)
    x = Concatenate(axis=-1)([x, topo_batch]) # concat with elevation (size= target size)
    x = conv1(x)
    x = conv2(x)
    x = Conv2D(filters=1,
               kernel_size=(5,5),
               strides=1,
               padding='same')(x)
    return tf.keras.Model(inputs=x0, outputs=x)

#===============================================================================================
#========================================= FSRCNN ==============================================
#===============================================================================================
def FSRCNNESM(xn, xm, ch, scale, aux = None): # Reconstructing High Resolution ESM Data Through a
                                       # Novel Fast Super Resolution Convolutional Neural Network

    # Paper says that elevation doesn't not improve performance, so not adding here.
    layers = []
    layers.append(MyLayers.CreateConv(filters=64, size=(5,5), strides=(1,1), apply_batchnorm=False)) # feature extraction
    layers.append(MyLayers.CreateConv(filters=32, size=(1,1), strides=(1,1), apply_batchnorm=False))# shrink
    layers.append(MyLayers.CreateConv(filters=12, size=(3,3), strides=(1,1), apply_batchnorm=False)) # mappings layers
    layers.append(MyLayers.CreateConv(filters=12, size=(3,3), strides=(1,1), apply_batchnorm=False)) # mappings layers
    layers.append(MyLayers.CreateDeconvUpsample(filters=1, size=(3,3), scale=scale, apply_batchnorm=False)) # upscaling
    layers.append(MyLayers.CreateConv(filters=64, size=(3,3), strides=(1,1), apply_batchnorm=False)) # patch extraction
    layers.append(MyLayers.CreateConv(filters=32, size=(3,3), strides=(1,1), apply_batchnorm=False)) # non-linear mapping

    x = tf.keras.Input(shape=(xn, xm, ch))
    x0 = x
    if(aux is not None):
        repeat_shape = tf.shape(x)
        topo_batch = tf.repeat(aux, repeat_shape[0], axis=0)

    for i, layer in enumerate(layers):
        if(aux is not None and i == 4): # after fed into DevoncUpsample layer
            x = layer(x)
            x = Concatenate(axis=-1)([x, topo_batch])
        else:
            x = layer(x)

    # Reconstruction
    x = Conv2D(filters=1,
               kernel_size=(1,1),
               strides=1,
               padding='same')(x)

    return tf.keras.Model(inputs=x0, outputs=x)

#===============================================================================================
#========================================= VGG-19 ==============================================
#===============================================================================================
'''ref: On the modern deep learning approaches for precipitation downscaling'''
def VGG(xn, xm, n_class=1):
    l2 = tf.keras.regularizers.L2(l2=5*1e-4)
    feature_extractor = []
    def BuildExtractor(n_layers, filters, kernel_size=(3,3), strides=(1,1),
                       pool_size=(2,2), pool_strides=(2,2)):
        seq = Sequential()
        for i in range(n_layers):
            seq.add(Conv2D(filters = filters, kernel_size = kernel_size,
                           strides = 1, padding='SAME', kernel_regularizer=l2))
            seq.add(Activation('relu'))
        seq.add(MaxPool2D(pool_size = pool_size, strides = pool_strides))
        return seq


    def BuildClassifier(n_layers, units, drop_rate, N_classes):
        assert(n_layers == len(units)), "Number of layers is not equal to number of nodes"
        seq = Sequential()
        seq.add(Dense(units=512, activation='relu', kernel_regularizer=l2))
        seq.add(Dropout(drop_rate, input_shape=(512,)))
        for i in range(n_layers-1):
            seq.add(Dense(units=units[i], activation='relu', kernel_regularizer=l2))
            seq.add(Dropout(drop_rate, input_shape=(units[i],)))
        seq.add(Dense(units=N_classes, activation='softmax', kernel_regularizer=l2))
        return seq
        
    feature_extractor.append(BuildExtractor(n_layers=2, filters=64))
    feature_extractor.append(BuildExtractor(n_layers=2, filters=128))
    feature_extractor.append(BuildExtractor(n_layers=4, filters=256))
    feature_extractor.append(BuildExtractor(n_layers=4, filters=512))
    feature_extractor.append(BuildExtractor(n_layers=4, filters=512))
    classifier = BuildClassifier(n_layers=3, units=[4096,1024,256],
                                 drop_rate=0.5, N_classes=n_class)
    x = tf.keras.Input(shape=(xn, xm, 1))
    x0 = x
    for layer in feature_extractor:
        x = layer(x)
    x = tf.keras.layers.Flatten(data_format='channels_last')(x)
    x = classifier(x)

    return tf.keras.Model(inputs=x0, outputs=x)