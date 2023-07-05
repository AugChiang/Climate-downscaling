from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, UpSampling2D, Activation, BatchNormalization, Input
from tensorflow.keras.layers import GlobalMaxPool2D, GlobalAveragePooling2D, MaxPool2D, Dense, Dropout, Lambda, Add
from tensorflow.keras.models import Sequential
import tensorflow as tf

def CreateConv(filters=64, size=(3,3), strides=1, apply_batchnorm=False):
    # initializer = tf.random_normal_initializer(0., 0.02)
    seq = Sequential()
    seq.add(Conv2D(filters, size, strides=strides, padding='same'))

    if(apply_batchnorm):
        seq.add(BatchNormalization())

    seq.add(Activation('relu'))
    return seq

def CreateDeconv(filters=64, output_ch=64, size=(3,3), apply_batchnorm=False, ADD_CONV_AFTER=True):
    # initializer = tf.random_normal_initializer(0., 0.02)
    seq = Sequential()
    seq.add(Conv2DTranspose(filters, size, strides=1, padding='SAME'))

    if(ADD_CONV_AFTER):
        seq.add(Activation('relu'))
        seq.add(Conv2D(filters=output_ch, kernel_size=size, strides=1, padding='SAME')) # purpose: remove chess-board effect

    if(apply_batchnorm):
        seq.add(BatchNormalization())

    seq.add(Activation('relu'))
    return seq

def CreateUpsample(filters=1, size=(3,3), scale=2, interpolation='bilinear', apply_batchnorm=False):
    seq = Sequential()
    seq.add(Conv2D(filters = filters, kernel_size = size,strides = 1,padding='SAME'))
    seq.add(Activation('relu'))
    seq.add(UpSampling2D(size=(scale, scale), interpolation=interpolation))
    seq.add(Conv2D(filters = filters, kernel_size = size, strides = 1, padding='SAME'))

    if(apply_batchnorm):
        seq.add(BatchNormalization())

    seq.add(Activation('relu'))
    return seq

def CreateSubPixelUpsample(size=(3,3), scale=2, apply_batchnorm=False):
    seq = Sequential()
    seq.add(Conv2D(filters = scale*scale, kernel_size = size, strides = 1, padding='SAME'))
    seq.add(Lambda(lambda x:tf.nn.depth_to_space(x, scale)))
    if(apply_batchnorm):
        seq.add(BatchNormalization())
    return seq

def CreateDeconvUpsample(filters=1, size=(3,3), scale=2, apply_batchnorm=False):
    seq = Sequential()
    seq.add(Conv2DTranspose(filters = filters, kernel_size = size, strides = scale, padding='SAME'))
    # seq.add(Conv2D(filters = filters, kernel_size = size, strides = 1, padding='SAME'))    

    if(apply_batchnorm):
        seq.add(BatchNormalization())

    seq.add(Activation('relu'))
    return seq

def CreateAuxConcat(filters=64, size=(3,3), output_ch=1, apply_batchnorm=False):
    seq = Sequential()
    seq.add(Conv2D(filters = filters, kernel_size = size,strides = 1,padding='SAME'))

    if(apply_batchnorm):
        seq.add(BatchNormalization())

    seq.add(Activation('relu'))
    seq.add(Conv2D(filters = output_ch, kernel_size = 1, strides = 1, padding='VALID'))

    if(apply_batchnorm):
        seq.add(BatchNormalization())

    return seq
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
    def MLP() -> tf.Tensor:
        hidden_layer_widths = int(mlp_nodes * reduction_ratio)
        seq = Sequential()
        seq.add(Dense(mlp_nodes))
        seq.add(Dense(hidden_layer_widths))
        seq.add(Dense(n_ch))
        return seq
    mlp = MLP()
    merged_p = mlp(max_p) + mlp(avg_p) #shared mlp
    # print("Merged shape: ", merged_p.shape)
    ch_atten = Activation('sigmoid')(merged_p)
    return ch_atten

def SpAttenBlock(input_x:tf.Tensor, ksize=5)-> tf.Tensor:
    ch_wise_maxpool = Lambda(lambda X:tf.nn.max_pool(X, ksize=(1, 1, 1, 1),
                                                        strides=(1, 1, 1, 1),
                                                        padding="VALID"))(input_x) # output -> (batch, H, W, 1)
    ch_wise_avgpool = tf.math.reduce_mean(input_x, axis=-1, keepdims=True) # output -> (batch, H, W, 1)
    def CONVD():
        seq = Sequential()
        seq.add(Conv2D(filters=1, kernel_size=(ksize, ksize), strides = (1,1), padding = 'same'))
        seq.add(Activation('sigmoid'))
        return seq
    res = Concatenate(axis=-1)([ch_wise_avgpool,ch_wise_maxpool])
    conv = CONVD()
    res = conv(res)
    return res

if __name__ == "__main__":
    print("Test Module OK.")