import tensorflow as tf
from model_utils.dataloader import MyDataset, GetTopology
from model_utils.MyModelsNonSeq import MyModelV2
from model_utils.train import *
from config import Config

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

def main():
    ##################################### Configuration ############################################
    config = Config('config.ini')
    y_n, y_m = config.input_height*config.scale, config.input_width*config.scale # y-height and y-width
    INPUT_SIZE = (config.input_height, config.input_width, config.num_channel) # multi-feature inputs
    OUTPUT_SIZE = (y_n, y_m, 1) # only precipitation

    ########################################## Metrics/Loss ###########################################
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    train_acc_metric = tf.keras.metrics.MeanSquaredError()
    val_acc_metric = tf.keras.metrics.MeanSquaredError()

    ##################################### Dataset Class ##########################################
    dataset = MyDataset(config=config,
                        x_size=INPUT_SIZE,
                        y_size=OUTPUT_SIZE)

    # intermediate predictions, can be modified into GradCam in "dataloader.py"
    # grad_cam = GetGradCAM(tp_path = config.inter_pred_dir, 
    #                     xn = config.input_height,
    #                     xm = config.input_width)

    #################################### topo normalization #########################################
    topo = GetTopology(y_n = y_n,
                    y_m = y_m,
                    topo_path = config.topo_dir,
                    use_log1p  = config.topo_use_log1p,
                    use_01    = config.use_01)

    ############################################# Model Class ###############################################
    model = MyModelV2(config = config,
                      aux    = topo)
    model.compile(optimizer=optimizer, loss=loss_fn)

    ###################################### Training Loop ########################################
    train = Train(model   = model,
                  dataset = dataset,
                  config  = config)

    train.start_train(optimizer        = optimizer,
                      loss_fn          = loss_fn,
                      train_acc_metric = train_acc_metric,
                      val_acc_metric   = val_acc_metric
                    )

if __name__ == '__main__':
    main()