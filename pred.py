from tensorflow import image
from model_utils import MyModels
from model_utils.MyModels import MyModelV2
import numpy as np
import configparser
import glob
import os

def TopoResizeNorm(h, w, topo):
  topo = image.resize(topo, [h,w], method=image.ResizeMethod.BICUBIC)
  norm_y = np.where(topo<0, 0, topo)
  norm_y = (norm_y-np.min(norm_y))/(np.max(norm_y)-np.min(norm_y)) # 0 ~ 1
  return norm_y

def TopoResizeNormLog(h, w, topo):
  topo = image.resize(topo, [h,w], method=image.ResizeMethod.BICUBIC)
  topo = np.where(topo<0, 0, topo)
  norm_y = np.log1p(topo)
  return norm_y

# normalize the inputs to be predicted
def InputNorm(input, use_log1p, use_01, tr_max, tr_min):
    if use_log1p:
        input = np.log1p(input)
        tr_max = np.log1p(tr_max)
    if use_01:
        input = (input-tr_min) / tr_max
    return input

# denormalize the prediction
def PredDenorm(pred, use_log1p, use_01, gt_max, gt_min):
    if use_log1p:
        gt_max = np.log1p(gt_max)
    if use_01:
        pred = pred*gt_max + gt_min
    if use_log1p:
        pred = np.expm1(pred)
    return pred


config = configparser.ConfigParser()
config.read('config.ini')
# prediction input and output dirs
predInputPath = sorted(glob.glob(config['path']['pred_input']))
predOutputPath = config['path']['pred_results']
if os.path.exists(predOutputPath) is not True:
    os.mkdir(predOutputPath)

# normalization
gt_max = config['normalization'].getfloat('ground_truth_max')
gt_min = config['normalization'].getfloat('ground_truth_min')
tr_max = config['normalization'].getfloat('training_max')
tr_min = config['normalization'].getfloat('training_min')

# model configs
xn = config['resolution'].getint('input_height')
xm = config['resolution'].getint('input_width')
topo_x = config['resolution'].getint('topo_height')
topo_y = config['resolution'].getint('topo_width')
scale = config['resolution'].getint('scale')
ch = config['resolution'].getint('channel')
N = config['model'].getint('number_of_main_layers')
upsample = config['model']['upsample']
skip_connection = config['model'].getboolean('skip_connection')
batch_norm = config['model'].getboolean('batch_normalization')
use_log1p = config['normalization'].getboolean('use_log1p')
use_01 = config['normalization'].getboolean('use_01')
# topo
topo_path = config['path']['topo_path']
topo = np.reshape(np.load(topo_path), (1, topo_x, topo_y, 1))
topo = TopoResizeNorm(h = xn*scale, w = xm*scale)

# load paths of the inputs for the predictions
model_weights = os.path.join(config['path']['save_dir'], 'variables')
pred_path = sorted(glob.glob(config['path']['pred_input']))
model = MyModelV2(n_layers=N, xn=xn, xm=xm, ch=ch, scale=scale, upsample=upsample,
                  use_elevation=True, aux = topo, use_skip_connect=skip_connection,
                  BatchNorm=batch_norm)
model.load_weights(model_weights)

for idx, pred_input in enumerate(predInputPath):
    # Shape of pred_input should be channel packed,
    # like (H*W, ch) or (H,W,ch) for each input piece.
    # Or you can adjust the shape for reshaping here.
    pred_input = np.load(pred_input)
    pred_input = InputNorm(pred_input, use_log1p=use_log1p, use_01=use_01, tr_max=tr_max, tr_min=tr_min)
    pred_input = np.reshape(pred_input, (1,xn,xm,ch))
    pred_output = np.squeeze(model.predict(pred_input))
    pred_output = PredDenorm(pred_output, use_log1p=use_log1p, use_01=use_01, gt_max=gt_max, gt_min=gt_min)
    # save as .npy file with the shape of (yn, ym)
    np.save(os.path.join(predOutputPath, f'pred{idx+1}'), pred_output)
