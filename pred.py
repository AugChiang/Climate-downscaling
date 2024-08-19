from config import Config
from model_utils.dataloader import GetTopology, MyDataset
from model_utils.MyModelsNonSeq import MyModelV2
from glob import glob
import os
import numpy as np

# denormalize the prediction
def pred_denorm(pred, use_log1p, use_01, gt_max, gt_min):
    if use_log1p:
        gt_max = np.log1p(gt_max)
    if use_01:
        pred = pred*gt_max + gt_min
    if use_log1p:
        pred = np.expm1(pred)
    return pred

def main():
    '''
    Inferencing Mode
    '''
    ##################################### Configuration ############################################
    config = Config('config.ini')
    y_n, y_m = config.input_height*config.scale, config.input_width*config.scale # y-height and y-width
    INPUT_SIZE = (config.input_height, config.input_width, config.num_channel) # multi-feature inputs
    OUTPUT_SIZE = (y_n, y_m, 1) # only precipitation

    ##################################### Dataset Class ##########################################
    dataset = MyDataset(config=config,
                        x_size=INPUT_SIZE,
                        y_size=OUTPUT_SIZE)
    
    #################################### topo normalization #########################################
    topo = GetTopology(y_n = y_n,
                    y_m = y_m,
                    topo_path = config.topo_dir,
                    use_log1p  = config.topo_use_log1p,
                    use_01    = config.use_01)

    ############################################# Model Class ###############################################
    model = MyModelV2(config = config,
                      aux    = topo)
    model.load_weights("pretrained_weights/variables").expect_partial()

    pred_file_paths = glob(os.path.join(config.pred_input_dir, '*.npy'))

    for step, (date, x) in enumerate(dataset.gen_pred_input(pred_file_paths)):
        pred = model.predict(x)
        pred = np.squeeze(pred)
        pred = pred_denorm(pred,
                           use_log1p = config.use_log1p,
                           use_01 = config.use_01,
                           gt_max = config.label_max,
                           gt_min = config.label_min)
        np.save(os.path.join(config.pred_save_dir, date), pred)
        print(f"Saving prediction on {date}")
        

if __name__ == '__main__':
    main()
