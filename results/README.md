This folder store the results after training, </br>
including the model weights related files: `checkpoint`, `variables.data-00000-of-00001` and `variables.index`, </br>
trianing losses: `losses.npy`, validation losses: `val_losses.npy`, time costs per epoch: `time_history.npy` (you can modify the names in `train.py` in line 144-146), </br>
and a sub-folder (default:`pred_epoch`) containing model predictions every **N** epochs (**N** is defined in the **training** section in `config.ini` as the variable name of **save_pred_every_epoch**) </br>
