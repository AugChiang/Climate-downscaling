import glob
import math
import random
from tensorflow import data, reshape
from tensorflow import image
import numpy as np
import os

def Scale01(arr, min=None, max=None):
    '''Min-Max scaler to [0,1]'''
    if(min is None):
        min = np.min(arr)
    if(max is None):
        max = np.max(arr)
    return (arr-min)/(max-min)

def ScaleMinus1to1(arr, mean=None):
    '''Min-Max scaler to [-1,1]'''
    if(mean is None):
        mean = np.mean(arr)
    min = np.min(arr)
    max = np.max(arr)
    return (arr-mean)/(max-min)

def GetTopology(topo_path,
                y_n,
                y_m,
                topo_yn=70,
                topo_ym=45,
                use_01=False,
                use_log1p=False):
    topo = np.load(topo_path)
    topo = np.reshape(topo, (1,topo_yn,topo_ym,1))
    topo = image.resize(topo, [y_n, y_m], method=image.ResizeMethod.BICUBIC).numpy()
    topo = np.where(topo<0, 0, topo)
    if(use_log1p):
        topo = np.log1p(topo)
    if(use_01):
        topo = (topo-np.min(topo))/(np.max(topo)-np.min(topo)) # 0 ~ 1
    return topo

# TODO: modularize intermediate input
def GetGradCAM(tp_path, auxtrpath, auxtags, xn=14, xm=9):
    '''
    # Input
        tp_path: example precipitation data for resulting prediction during model training
        auxtrpath: auxlilary data path lists in the form of [folder1, folder2, ... ]
        auxtags: auxlilary data type, correseponding to 'auxtrpath' order in the form of [tag1, tag2, ... ]
    # Output
        reshaped (4D) and normalized to [0,1] by their min-max value of total data
        for checking resulting prediction during training.
    '''
    grad_cam = np.load(tp_path)
    grad_cam = np.log1p(grad_cam)
    grad_cam = Scale01(arr=grad_cam, min=0, max=np.log1p(31.544506))
    ch = 1
    if(auxtrpath is not None and auxtags is not None):
        ch += len(auxtags)
        for i in range(len(auxtrpath)):
            aux_path = str(auxtrpath[i] + f'/{auxtags[i]}_20120612.npy') # aux npy file path
            aux = np.load(aux_path)
            # aux dat normalization
            if(auxtags[i] == 'u'): # scale wind components by aux type total min max
                # aux = ScaleMinus1to1(aux, mean=1.5) # [-1,1]
                aux = Scale01(aux, min=-19.8128, max=22.79312) # [0,1]
                grad_cam = np.hstack((grad_cam, aux))
            elif(auxtags[i] == 'v'): # scale wind components by aux type total min max
                # aux = ScaleMinus1to1(aux, mean=-0.27) # to [-1,+1]
                aux = Scale01(aux, min=-23.51668, max=23.240267) # [0,1]
                grad_cam = np.hstack((grad_cam, aux))
            elif(auxtags[i] == 't2m'): # scale humidity to [0,+1] by aux type total min max
                aux = Scale01(aux, min=-5.2, max=32.262) # scale other aux data to [0,1] unit:Celcius
                grad_cam = np.hstack((grad_cam, aux))
            elif(auxtags[i] == 'msl'): # scale humidity to [0,+1] by aux type total min max
                aux = Scale01(aux, min=0, max=0.0146) # scale other aux data to [0,1]
                grad_cam = np.hstack((grad_cam, aux))
            elif(auxtags[i] == 'q700'): # scale sea-level pressure to [0,+1] by aux type total min max
                aux = Scale01(aux, min=96824.2, max=103687.2) # scale other aux data to [0,1]
                grad_cam = np.hstack((grad_cam, aux))
    grad_cam = np.reshape(grad_cam, (1, xn, xm, ch))
    return grad_cam

class MyDataset():
    def __init__(self,
                 config:object,
                 x_size,
                 y_size,
                 xpatch_size=None,
                 ypatch_size=None,
                 auxtrpath=None,
                 auxtags=None,
                 split=0.8):
        '''
        xtrpath: training data path folder address.
        ytrpath: ground truth  path folder address.
        auxtrpath: aux data path folder address. (type:list)
        xsize: x data (H,W,C)
        ysize: y data (H,W,C)
        patch size: for slicing into patched image (square size).
        split: ratio of splitting dataset into training and remained ratio for testing.
               default: split whole dataset into 8:1:1 of training, validation, testing.
        use log1: data = log(data + 1)
        '''
        self.config = config
        # for attr in ['x', 'y']:
        #     setattr(self, attr, getattr(config, attr))
        self.xtrpath = sorted(glob.glob(os.path.join(self.config.input_dir, '*.npy')))
        self.ytrpath = sorted(glob.glob(os.path.join(self.config.label_dir, '*.npy')))
        self.path_list = list(zip(self.xtrpath, self.ytrpath))
        self.auxtrpath = auxtrpath  # root dir of aux .npy files
        self.scale = self.config.scale
        self.auxtags = auxtags      # ['t2m', 'u', 'v', 'q700', 'msl'...] for .npy file name to stack with tp
        self.x_size = x_size
        self.y_size = y_size
        
        self.xpatch_size = xpatch_size
        self.ypatch_size = ypatch_size
        self.split = split

        # normalization constant
        self.max = {"input"  :np.log1p(config.input_max),
                    "scale2" :np.log1p(config.label_max_x2),
                    "scale4" :np.log1p(config.label_max_x4),
                    "scale5" :np.log1p(config.label_max),
                    "scale8" :np.log1p(config.label_max_x8),
                    "scale25":np.log1p(config.label_max_x25)}

        self.len = len(self.xtrpath)

    def _path_check(self):
        if(len(self.xtrpath) != len(self.ytrpath)):
            print("The number of paths is not the same.")
            return False
        else:
            for pathi in range(len(self.xtrpath)):
                # TODO: check file with correseponding file name might be problematic
                if(self.xtrpath[pathi][-12:] != self.ytrpath[pathi][-12:]): # check if the same yyyymmdd
                    print("The corresponding paths of training and truth are not alike.")
                    return False
            return True
    
    def _get_data_path(self):
        '''
        Shuffle and split the path_list into training, validation, and test data paths.
        '''
        if(self._path_check):
            if(self.config.random_seed is not None):
                random.Random(self.config.random_seed).shuffle(self.path_list) # shuffle the paths with seed
            train_len = math.floor(len(self.path_list)*self.split)
            test_len = math.floor((len(self.path_list) - train_len)/2)
            # train, val, test
            return self.path_list[:train_len], self.path_list[train_len:train_len+test_len], self.path_list[train_len+test_len:]
        return False

    def _gen_data(self, paths):
        '''
        # Input
        paths: list of tuples, [(x0, y0), (x1, y1)...] paired paths

        # Output
        Preprocessed x,y data.
        If multi-channel is True, x is multi-channel concatenated.
        Multi-channel is configured in class.__init__() of auxtrpath and auxtags.
        '''

        for i in range(len(paths)):
            date = paths[i][0][-12:].decode('utf-8') # yyyymmdd.npy
            # print("Date: ", date) # type:=bytes
            x = np.load(paths[i][0])
            y = np.load(paths[i][1])

            if(self.config.use_log1p): # x' = ln(x+1) for normalization
                x = np.where(x<0, 0, x)
                y = np.where(y<0, 0, y)
                x = np.log1p(x)
                y = np.log1p(y)

            if(self.config.use_01): # normalize to [0,1] 
                # x = (x-np.min(x)/(np.max(x)-np.min(x)+1e-4))
                # y = (y-np.min(y)/(np.max(y)-np.min(y)+1e-4))
                # by total min-max = [0,31.544506]
                x = Scale01(arr=x, min=0, max=self.max['input'])
                y = Scale01(arr=y, min=0, max=self.max[f'scale{self.scale}'])
            
            # TODO: pull aux stat into config file and read here
            # concatenate aux data as additional channel of precipitation data
            if(self.auxtrpath is not None):
                for i in range(len(self.auxtrpath)):
                    aux_path = str(self.auxtrpath[i] + '/' + self.auxtags[i] + '_' + date) # aux npy file path
                    aux = np.load(aux_path)
                    # aux dat normalization
                    if(self.auxtags[i] == 't2m'): # scale humidity to [0,+1] by aux type total min max
                        aux = Scale01(aux, min=-5.2, max=32.262) # scale other aux data to [0,1] unit:Celcius
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'u'): # scale wind components to [0,+1] by its total min max
                        # aux = ScaleMinus1to1(aux, mean=1.5) # to [-1,1]
                        aux = Scale01(aux, min=-19.8128, max=22.79312)
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'v'): # scale wind components to [0,+1] by its total min max
                        # aux = ScaleMinus1to1(aux, mean=-0.27) # to [-1,1]
                        aux = Scale01(aux, min=-23.51668, max=23.240267)
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'msl'): # scale humidity to [0,+1] by its total min max
                        aux = Scale01(aux, min=0, max=0.0146) # scale other aux data to [0,1]
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'q700'): # scale sea-level pressure to [0,+1] by its total min max
                        aux = Scale01(aux, min=96824.2, max=103687.2) # scale other aux data to [0,1]
                        x = np.hstack((x, aux))

            x = np.reshape(x, self.x_size)
            y = np.reshape(y, self.y_size)
            yield x, y

    def _gen_patch_data(self, paths):
        for i in range(len(paths)):
            # 0 | 1
            # -----
            # 2 | 3
            xcut = i%4
            x = np.load(paths[i][0])
            y = np.load(paths[i][1])

            x = np.where(x<0, 0, x)
            y = np.where(y<0, 0, y)

            if(self.config.use_log1p):
                x = np.log1p(x)
                y = np.log1p(y)
            x = np.reshape(x, (100,60))
            y = np.reshape(y, (200,120))

            x = x[50*(xcut//2):50*(xcut//2)+50,
                  30*(xcut%2):30*(xcut%2)+30]
            y = y[100*(xcut//2):100*(xcut//2)+100,
                  60*(xcut%2):60*(xcut%2)+60]
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)
            yield x, y

    def _gen_vgg_data(self,paths):
        for i in range(len(paths)):
            x = np.load(paths[i][0])
            x = np.where(x<0, 0, x)
            x = (x-np.min(x)/(np.max(x)-np.min(x)+1e-4))
            x = np.reshape(x, self.x_size)
            yield x

    def train_dataset_gen(self):
        '''
        Generate a batch of paired training data in the form of (x_train, y_train)
        '''
        trp, _, __ = self._get_data_path()
        train_dataset = data.Dataset.from_generator(
                self._gen_data, args = [trp],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        train_dataset = train_dataset.batch(self.config.batch_size) # 4D
        return train_dataset

    def val_dataset_gen(self):
        '''
        Generate a batch of paired validation data in the form of (x_val, y_val)
        '''
        _, val, __ = self._get_data_path()
        val_dataset = data.Dataset.from_generator(
                self._gen_data, args = [val],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        val_dataset = val_dataset.batch(self.config.batch_size)
        return val_dataset

    def test_dataset_gen(self):
        '''
        Generate a batch of paired test data in the form of (x_test, y_test)
        '''
        _, __, test = self._get_data_path()
        test_dataset = data.Dataset.from_generator(
                self._gen_data, args = [test],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        test_dataset = test_dataset.batch(self.config.batch_size)
        return test_dataset

    def patch_train_dataset_gen(self):
        trp, _, __ = self._get_data_path()
        train_dataset = data.Dataset.from_generator(
                self._gen_patch_data, args = [trp],
                output_types = (np.float32, np.float32), 
                output_shapes = (self.xpatch_size, self.ypatch_size))
        train_dataset = train_dataset.batch(self.config.batch_size) # 4D
        return train_dataset

    def patch_val_dataset_gen(self):
        _, val, __ = self._get_data_path()
        val_dataset = data.Dataset.from_generator(
                self._gen_patch_data, args = [val],
                output_types = (np.float32, np.float32),
                output_shapes = (self.xpatch_size, self.ypatch_size))
        val_dataset = val_dataset.batch(self.config.batch_size) # 4D
        return val_dataset

    def patch_test_dataset_gen(self):
        _, __, test = self._get_data_path()
        test_dataset = data.Dataset.from_generator(
                self._gen_patch_data, args = [test],
                output_types = (np.float32, np.float32),
                output_shapes = (self.xpatch_size, self.ypatch_size))
        test_dataset = test_dataset.batch(self.config.batch_size) # 4D
        return test_dataset
    
    def vgg_train_dataset_gen(self):
        trp, _, __ = self._get_data_path()
        train_dataset = data.Dataset.from_generator(
                self._gen_vgg_data, args = [trp],
                output_types = (np.float32),
                output_shapes = (self.xpatch_size))
        train_dataset = train_dataset.batch(self.config.batch_size) # 4D
        return train_dataset

    def vgg_val_dataset_gen(self):
        _, val, __ = self._get_data_path()
        val_dataset = data.Dataset.from_generator(
                self._gen_vgg_data, args = [val],
                output_types = (np.float32),
                output_shapes = (self.xpatch_size))
        val_dataset = val_dataset.batch(self.config.batch_size) # 4D
        return val_dataset

    def gen_pred_input(self, paths):
        '''
        Generator generates date and array.
        Args:
            paths (Iterable): data path of input data for inferencing.

        Yields:
            _type_: date (str) and array (default to numpy array)
        '''
        for i in range(len(paths)):
            date = paths[i][-12:] # yyyymmdd.npy
            # print("Date: ", date) # type:=bytes
            x = np.load(paths[i])

            if(self.config.use_log1p): # x' = ln(x+1) for normalization
                x = np.where(x<0, 0, x)
                x = np.log1p(x)

            if(self.config.use_01): # normalize to [0,1] 
                x = Scale01(arr=x, min=0, max=self.max['input'])
            
            # TODO: pull aux stat into config file and read here
            # concatenate aux data as additional channel of precipitation data
            if(self.auxtrpath is not None):
                for i in range(len(self.auxtrpath)):
                    aux_path = str(self.auxtrpath[i] + '/' + self.auxtags[i] + '_' + date) # aux npy file path
                    aux = np.load(aux_path)
                    # aux dat normalization
                    if(self.auxtags[i] == 't2m'): # scale humidity to [0,+1] by aux type total min max
                        aux = Scale01(aux, min=-5.2, max=32.262) # scale other aux data to [0,1] unit:Celcius
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'u'): # scale wind components to [0,+1] by its total min max
                        # aux = ScaleMinus1to1(aux, mean=1.5) # to [-1,1]
                        aux = Scale01(aux, min=-19.8128, max=22.79312)
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'v'): # scale wind components to [0,+1] by its total min max
                        # aux = ScaleMinus1to1(aux, mean=-0.27) # to [-1,1]
                        aux = Scale01(aux, min=-23.51668, max=23.240267)
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'msl'): # scale humidity to [0,+1] by its total min max
                        aux = Scale01(aux, min=0, max=0.0146) # scale other aux data to [0,1]
                        x = np.hstack((x, aux))
                    if(self.auxtags[i] == 'q700'): # scale sea-level pressure to [0,+1] by its total min max
                        aux = Scale01(aux, min=96824.2, max=103687.2) # scale other aux data to [0,1]
                        x = np.hstack((x, aux))

            x = np.reshape(x, self.x_size)
            x = x[np.newaxis, ...]
            yield date, x

if __name__ == '__main__' :
    print("Test Module OK.")
