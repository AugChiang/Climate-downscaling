import tensorflow as tf
from model_utils.dataloader import MyDataset, GetTopology, GetGradCAM
from model_utils.MyModels import MyModelV2
from model_utils.train import *
import numpy as np
import configparser
import math

##################################### Model Summary ########################################
'''
dataset: hetergeneous, size = 14x9 to size = 70x45 (scaling factor = 5)
         single channels: precipitation
total number of layers = 32
topo: 01 normalization
upscaling layer = subpixel
input skip connection = True
batch = 64
epochs = 1000
shuffling seed of training data = 1
patience = 60
loss = MSE
'''
##################################### Configuration ############################################
config = configparser.ConfigParser()
config.read('config.ini')
x_n = config['resolution'].getint('x_n') # x-height
x_m = config['resolution'].getint('x_m') # x-width
SCALE = config['resolution'].getint('scale') # scaling factor
ch = config['resolution'].getint('channel') # number of channels
y_n, y_m = x_n*SCALE, x_m*SCALE # y-height and y-width
INPUT_SIZE = (x_n, x_m, ch) # multi-feature inputs
OUTPUT_SIZE = (y_n, y_m, 1) # only precipitation

USE_LOG = config['normalization'].getboolean('use_log1p')
TOPO_USE_LOG = config['normalization'].getboolean('topo_use_log1p')
USE_01 = config['normalization'].getboolean('use_01')

SKIP_CONNECT = config['model'].getboolean('skip_connection')
BATCH_NORM = config['model'].getboolean('batch_normalization')
N = config['model'].getint('number_of_main_layers')
UPSAMPLE = config['model']['upsample']

BATCH_SIZE = config['training'].getint('batch_size')
EPOCHS = config['training'].getint('epochs')
SAVE_EVERY_EPOCH = config['training'].getint('save_pred_every_epoch')
SEED = config['training'].getint('random_seed')
patience = config['training'].getint('patience')
lr = config['training'].getfloat('learning_rate')


########################################## Metrics/Loss ###########################################
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
train_acc_metric = tf.keras.metrics.MeanSquaredError()
val_acc_metric = tf.keras.metrics.MeanSquaredError()

##################################### Paths ############################################
x_train_path = config['path']['x_training_path']
y_train_path = config['path']['y_training_path']
x_ele_path = config['path']['topo_path']
inter_eg = x_train_path + config['path']['gradcam_path']
SaveName = config['path']['results']
auxtrpath = None
auxtags = None

##################################### Dataset Class ##########################################
dataset = MyDataset(xtrpath=x_train_path, ytrpath=y_train_path,
                    auxtrpath=auxtrpath, auxtags = auxtags,
                    x_size=INPUT_SIZE, y_size=OUTPUT_SIZE,
                    batch_size=BATCH_SIZE, seed=SEED,
                    use_log1=USE_LOG,  use_01=USE_01)

# intermediate predictions, can be modified into GradCam in "dataloader.py"
grad_cam = GetGradCAM(tp_path = inter_eg, 
                      auxtrpath = auxtrpath,
                      auxtags = auxtags,
                      xn = x_n, xm = x_m)

##################################### File Paths ############################################
# standard setting: total number of data ~= batch_size * epoch
TOTAL_DATA_NUM = dataset.len
TOTAL_TEST_NUM = math.floor(TOTAL_DATA_NUM*(1-dataset.split)*0.5)
ITERS = int(TOTAL_DATA_NUM * EPOCHS / BATCH_SIZE)

#################################### topo normalization #########################################
topo = GetTopology(topo_path=x_ele_path,
                   y_n=y_n, y_m=y_m,
                   use_log1=TOPO_USE_LOG,
                   use_01=USE_01)

############################################# Model Class ###############################################
model = MyModelV2(n_layers=N, xn=x_n, xm=x_m, ch=ch, scale=SCALE, Upsample=UPSAMPLE,
                  use_elevation=True, aux = topo, use_skip_connect=SKIP_CONNECT, BatchNorm=BATCH_NORM)
model.compile(optimizer=optimizer, loss=loss_fn)
# print(model.summary())
# tf.keras.utils.plot_model(model, to_file=f"{SaveName}.png", show_shapes=True, dpi=64)
###################################### Training Loop ########################################
train = Train(model = model, save_folder_name = SaveName,
              x_n = x_n, x_m = x_m,
              dataset = dataset, grad_cam = grad_cam)

train.start_train(optimizer=optimizer, loss_fn=loss_fn,
                  train_acc_metric=train_acc_metric,
                  val_acc_metric=val_acc_metric,
                  epochs=EPOCHS, save_pred_every_epoch=SAVE_EVERY_EPOCH, patience=patience)
