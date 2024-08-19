from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, UpSampling2D, Activation, BatchNormalization, Input
from tensorflow.keras.layers import GlobalMaxPool2D, GlobalAveragePooling2D, MaxPool2D, Dense, Dropout, Lambda, Add
from tensorflow.keras.models import Sequential
import tensorflow as tf

def CreateConv(input, filters=64, size=(3,3), strides=1, apply_batchnorm=False):
    # initializer = tf.random_normal_initializer(0., 0.02)
    y = Conv2D(filters, size, strides=strides, padding='same')(input)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    y = Activation('relu')(y)
    return y

def CreateDeconv(input, filters=64, output_ch=64, size=(3,3), apply_batchnorm=False, ADD_CONV_AFTER=True):
    # initializer = tf.random_normal_initializer(0., 0.02)
    
    y = Conv2DTranspose(filters, size, strides=1, padding='SAME')(input)

    if(ADD_CONV_AFTER):
        y = Activation('relu')(y)
        y = Conv2D(filters=output_ch, kernel_size=size, strides=1, padding='SAME')(y) # purpose: remove chess-board effect

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    y = Activation('relu')(y)
    return y

def CreateUpsample(input, filters=1, size=(3,3), scale=2, interpolation='bilinear', apply_batchnorm=False):
    y = Conv2D(filters = filters, kernel_size = size,strides = 1,padding='SAME')(input)
    y = Activation('relu')(y)
    y = UpSampling2D(size=(scale, scale), interpolation=interpolation)(y)
    y = Conv2D(filters = filters, kernel_size = size, strides = 1, padding='SAME')(y)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    y = Activation('relu')(y)
    return y

def CreateSubPixelUpsample(input, size=(3,3), scale=2, apply_batchnorm=False):
    y = Conv2D(filters = scale*scale, kernel_size = size, strides = 1, padding='SAME')(input)
    y = Lambda(lambda x:tf.nn.depth_to_space(x, scale))(y)
    if(apply_batchnorm):
        y = BatchNormalization()(y)
    return y

def CreateDeconvUpsample(input, filters=1, size=(3,3), scale=2, apply_batchnorm=False):
    y = Conv2DTranspose(filters = filters, kernel_size = size, strides = scale, padding='SAME')(input)
    # y = Conv2D(filters = filters, kernel_size = size, strides = 1, padding='SAME')(y)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    y = Activation('relu')(y)
    return y

def CreateAuxConcat(input, filters=64, size=(3,3), output_ch=1, apply_batchnorm=False):
    y = Conv2D(filters = filters, kernel_size = size,strides = 1,padding='SAME')(input)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = Conv2D(filters = output_ch, kernel_size = 1, strides = 1, padding='SAME')(y)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    return y

def CreateAuxConcatV1(input, filters=64, size=(3,3), output_ch=1, apply_batchnorm=False):
    y = Conv2D(filters = filters, kernel_size = size, strides = 1,padding='SAME')(input)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = Conv2D(filters = output_ch, kernel_size = size, strides = 1, padding='SAME')(y)

    if(apply_batchnorm):
        y = BatchNormalization()(y)

    return y
#=====================================================================================
def ResidualDenseBlock(input_x:tf.Tensor, n_layers:int, filters:int,
                       size=(3,3), strides=1, apply_batchnorm=False) -> tf.Tensor:
    res = [input_x]

    x = Conv2D(filters = filters, kernel_size = size,
               strides = strides, padding='SAME')(input_x)
    x = Activation('relu')(x)
     # tensor for connection to next node

    for i in range(n_layers-1):
        xr = Concatenate(axis=-1)([r for r in res])
        x = Concatenate(axis=-1)([x, xr])
        x = Conv2D(filters = filters, kernel_size = size,
                   strides = strides, padding='SAME')(x)
        x = Activation('relu')(x)
        res.append(x) # tensor for connection to next node

    x = Concatenate(axis=-1)([r for r in res])
    x = Conv2D(filters = 1, kernel_size = size,
               strides = strides, padding='SAME')(x)
    x = x + input_x
    res = []
    return x
#=====================================================================================
def ChAttenBlock(input_x:tf.Tensor, mlp_nodes=256, reduction_ratio = 0.5) -> tf.Tensor:
    # H = input_x.shape[1]
    # W = input_x.shape[2]
    n_ch = input_x.shape[-1]
    max_p = GlobalMaxPool2D(data_format='channels_last',
                            keepdims=True)(input_x) # output -> (batch, 1, 1, channel)
    # print("Shape after MaxP: ", max_p.shape)
    avg_p = GlobalAveragePooling2D(data_format='channels_last',
                                   keepdims=True)(input_x) # output -> (batch, 1, 1, channel)
    # print("Shape after AvgP: ", avg_p.shape)
    '''https://stackoverflow.com/a/52092176'''
    def MLP(input) -> tf.Tensor:
        hidden_layer_widths = int(mlp_nodes * reduction_ratio)
        y = Dense(mlp_nodes)(input)
        y = Dense(hidden_layer_widths)(y)
        y = Dense(n_ch)(y)
        return y
    merged_p = MLP(max_p) + MLP(avg_p) #shared mlp
    # print("Merged shape: ", merged_p.shape)
    ch_atten = Activation('sigmoid')(merged_p)
    return ch_atten

def SpAttenBlock(input_x:tf.Tensor, ksize=5)-> tf.Tensor:
    ch_wise_maxpool = Lambda(lambda X:tf.nn.max_pool(X, ksize=(1, 1, 1, 1),
                                                        strides=(1, 1, 1, 1),
                                                        padding="VALID"))(input_x) # output -> (batch, H, W, 1)
    ch_wise_avgpool = tf.math.reduce_mean(input_x, axis=-1, keepdims=True) # output -> (batch, H, W, 1)
    def CONVD(input):
      y = Conv2D(filters=1, kernel_size=(ksize, ksize), strides = (1,1), padding = 'same')(input)
      y = Activation('sigmoid')(y)
      return y
    res = Concatenate(axis=-1)([ch_wise_avgpool,ch_wise_maxpool])
    res = CONVD(res)
    return res