import configparser

class Config():
    def __init__(self, config_path="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self._parse_config()

    def _parse_config(self):
        # dir section
        self.input_dir = self.config.get('dir', 'input_dir')
        self.label_dir = self.config.get('dir', 'label_dir')
        self.pred_input_dir = self.config.get('dir', 'pred_input_dir')
        self.pred_save_dir = self.config.get('dir', 'pred_save_dir')
        self.topo_dir = self.config.get('dir', 'topo_dir')
        self.inter_pred_dir = self.config.get('dir', 'inter_pred_dir')
        self.history_save_dir = self.config.get('dir', 'history_save_dir')
        self.model_save_dir = self.config.get('dir', 'model_save_dir')
        
        # Shape
        self.input_height = self.config.getint('shape', 'input_height')
        self.input_width = self.config.getint('shape', 'input_width')
        self.scale = self.config.getint('shape', 'scale')
        self.num_channel = self.config.getint('shape', 'num_channel')
        self.toop_height = self.config.getint('shape', 'topo_height')
        self.toop_height = self.config.getint('shape', 'topo_width')
    
        # Normalization
        self.use_log1p = self.config.getboolean('normalization','use_log1p')
        self.topo_use_log1p = self.config.getboolean('normalization','topo_use_log1p')
        self.use_01 = self.config.getboolean('normalization','use_01')
        self.input_max = self.config.getfloat('normalization','input_max')
        self.input_min = self.config.getfloat('normalization','input_min')
        self.label_max = self.config.getfloat('normalization','label_max')
        self.label_min = self.config.getfloat('normalization','label_min')
        self.label_max_x2 = self.config.getfloat('normalization','label_max_x2')
        self.label_max_x4 = self.config.getfloat('normalization','label_max_x4')
        self.label_max_x8 = self.config.getfloat('normalization','label_max_x8')
        self.label_max_x25 = self.config.getfloat('normalization','label_max_x25')
        
        # model
        self.skip_connection = self.config.getboolean('model','skip_connection')
        self.batch_norm = self.config.getboolean('model','batch_norm')
        self.num_main_layers = self.config.getint('model','num_main_layers')
        self.upsample = self.config.get('model','upsample')
        
        # training
        self.batch_size = self.config.getint('hyperparams', 'batch_size')
        self.epochs = self.config.getint('hyperparams', 'epochs')
        self.save_every_n_epoch = self.config.getint('hyperparams', 'save_every_n_epoch')
        self.random_seed = self.config.getint('hyperparams', 'random_seed')
        self.patience = self.config.getfloat('hyperparams', 'patience')
        self.learning_rate = self.config.getfloat('hyperparams', 'learning_rate')
