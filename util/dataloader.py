import glob
import math
import random
from tensorflow import data, reshape
from tensorflow import image
import numpy as np

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

def GetTopology(topo_path, y_n, y_m, use_01=False, use_log1=False):
    topo = np.load(topo_path)
    topo = np.reshape(topo, (1,400,240,1))
    topo = image.resize(topo, [y_n, y_m], method=image.ResizeMethod.BICUBIC).numpy()
    topo = np.where(topo<0, 0, topo)
    if(use_log1):
        topo = np.log1p(topo)
    if(use_01):
        topo = (topo-np.min(topo))/(np.max(topo)-np.min(topo)) # 0 ~ 1
    return topo

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
    def __init__(self, xtrpath, ytrpath, x_size, y_size,
                 batch_size, xpatch_size=None, ypatch_size=None, auxtrpath=None, auxtags=None,
                 seed=None, split=0.8, use_log1 = False, use_01 = False):
        '''
        xtrpath: training data path folder address.
        ytrpath: ground truth  path folder address.
        auxtrpath: aux data path folder address. (type:list)
        xsize: x data (H,W,C)
        ysize: y data (H,W,C)
        seed: shuffling seed.
        patch size: for slicing into patched image (square size).
        split: ratio of splitting dataset into training and remained ratio for testing.
               default: split whole dataset into 8:1:1 of training, validation, testing.
        use log1: data = log(data + 1)
        '''
        self.xtrpath = sorted(glob.glob(xtrpath + '/*.npy'))
        self.ytrpath = sorted(glob.glob(ytrpath + '/*.npy'))
        self.path_list = list(zip(self.xtrpath, self.ytrpath))
        self.auxtrpath = auxtrpath  # root dir of aux .npy files
        self.auxtags = auxtags      # ['t2m', 'u', 'v', 'q700', 'msl'...] for .npy file name to stack with tp
        self.x_size = x_size
        self.y_size = y_size
        self.scale = y_size[0] // x_size[0]
        self.xpatch_size = xpatch_size
        self.ypatch_size = ypatch_size

        self.batch_size = batch_size
        self.seed = seed
        self.split = split
        self.use_log1 = use_log1
        self.use_01 = use_01
        # normalization constant
        self.max = {"input":np.log1p(31.544506), "scale2":np.log1p(1341.9022), "scale4":np.log1p(1406.7449),
                    "scale5":np.log1p(1401.9225), "scale8":np.log1p(1517.90),"scale25":np.log1p(1609.3981)}

        self.len = len(self.xtrpath)
    
    def __pathcheck__(self):
        if(len(self.xtrpath) != len(self.ytrpath)):
            print("The number of paths is not the same.")
            return False
        else:
            for pathi in range(len(self.xtrpath)):
                if(self.xtrpath[pathi][-12:] != self.ytrpath[pathi][-12:]):
                    print("The corresponding paths of training and truth are not alike.")
                    return False
            return True
    
    def __getdatapath__(self):
        '''
        Shuffle and split the path_list into training, validation, and test data paths.
        '''
        if(self.__pathcheck__):
            if(self.seed is not None):
                random.Random(self.seed).shuffle(self.path_list) # shuffle the paths with seed
            train_len = math.floor(len(self.path_list)*self.split)
            test_len = math.floor((len(self.path_list) - train_len)/2)
            # train, val, test
            return self.path_list[:train_len], self.path_list[train_len:train_len+test_len], self.path_list[train_len+test_len:]
        return False

    def __datagen__(self, paths):
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

            if(self.use_log1): # x' = ln(x+1) for normalization
                x = np.where(x<0, 0, x)
                y = np.where(y<0, 0, y)
                x = np.log1p(x)
                y = np.log1p(y)

            if(self.use_01): # normalize to [0,1] 
                # x = (x-np.min(x)/(np.max(x)-np.min(x)+1e-4))
                # y = (y-np.min(y)/(np.max(y)-np.min(y)+1e-4))
                # by total min-max = [0,31.544506]
                x = Scale01(arr=x, min=0, max=self.max['input'])
                y = Scale01(arr=y, min=0, max=self.max[f'scale{self.scale}'])
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

    def __patchdatagen__(self, paths):
        for i in range(len(paths)):
            # 0 | 1
            # -----
            # 2 | 3
            xcut = i%4
            x = np.load(paths[i][0])
            y = np.load(paths[i][1])

            x = np.where(x<0, 0, x)
            y = np.where(y<0, 0, y)

            if(self.use_log1):
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

    def __vggdatagen__(self,paths):
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
        trp, _, __ = self.__getdatapath__()
        train_dataset = data.Dataset.from_generator(
                self.__datagen__, args = [trp],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        train_dataset = train_dataset.batch(self.batch_size) # 4D
        return train_dataset

    def val_dataset_gen(self):
        '''
        Generate a batch of paired validation data in the form of (x_val, y_val)
        '''
        _, val, __ = self.__getdatapath__()
        val_dataset = data.Dataset.from_generator(
                self.__datagen__, args = [val],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        val_dataset = val_dataset.batch(self.batch_size)
        return val_dataset

    def test_dataset_gen(self):
        '''
        Generate a batch of paired test data in the form of (x_test, y_test)
        '''
        _, __, test = self.__getdatapath__()
        val_dataset = data.Dataset.from_generator(
                self.__datagen__, args = [test],
                output_types = (np.float32, np.float32),
                output_shapes = (self.x_size, self.y_size))
        val_dataset = val_dataset.batch(self.batch_size)
        return val_dataset

    def patch_train_dataset_gen(self):
        trp, _, __ = self.__getdatapath__()
        train_dataset = data.Dataset.from_generator(
                self.__patchdatagen__, args = [trp],
                output_types = (np.float32, np.float32), 
                output_shapes = (self.xpatch_size, self.ypatch_size))
        train_dataset = train_dataset.batch(self.batch_size) # 4D
        return train_dataset

    def patch_val_dataset_gen(self):
        _, val, __ = self.__getdatapath__()
        val_dataset = data.Dataset.from_generator(
                self.__patchdatagen__, args = [val],
                output_types = (np.float32, np.float32),
                output_shapes = (self.xpatch_size, self.ypatch_size))
        val_dataset = val_dataset.batch(self.batch_size) # 4D
        return val_dataset

    def patch_test_dataset_gen(self):
        _, __, test = self.__getdatapath__()
        test_dataset = data.Dataset.from_generator(
                self.__patchdatagen__, args = [test],
                output_types = (np.float32, np.float32),
                output_shapes = (self.xpatch_size, self.ypatch_size))
        test_dataset = test_dataset.batch(self.batch_size) # 4D
        return test_dataset
    
    def vgg_train_dataset_gen(self):
        trp, _, __ = self.__getdatapath__()
        train_dataset = data.Dataset.from_generator(
                self.__vggdatagen__, args = [trp],
                output_types = (np.float32),
                output_shapes = (self.xpatch_size))
        train_dataset = train_dataset.batch(self.batch_size) # 4D
        return train_dataset

    def vgg_val_dataset_gen(self):
        _, val, __ = self.__getdatapath__()
        val_dataset = data.Dataset.from_generator(
                self.__vggdatagen__, args = [val],
                output_types = (np.float32),
                output_shapes = (self.xpatch_size))
        val_dataset = val_dataset.batch(self.batch_size) # 4D
        return val_dataset

if __name__ == '__main__' :
    print("Test Module OK.")