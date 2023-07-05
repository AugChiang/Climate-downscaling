# Climate-downscaling
Deep-learning Based Super-resolution Model in Climate Downscaling of Precipitation Data </br>
National Taiwan Normal University 110 sememster </br>
Advisor: Ko Chih Wang </br>
Author: Chia Hao Chiang </br>

## Purpose
To generate high-resolution precipitation data in Taiwan island </br>
Input: ERA5 Reanalysis Data </br>
Ground Truth: TCCIP Observations </br>

## Usage
### setting
Set your own parameters in "config.ini" </br>
Put your training dataset in the folder: "training_dataset," which provides '.npy' files </br>
Similarly, put your ground truth dataset in the folder: "ground_truth," which is also '.npy' files </br>
The file names of training data pieces and its corresponding ground truth pieces in default are set to be "tp_{yyyymmdd}.npy" and "{yyyymmdd}.npy," respectively </br>

### training
Direct to the root folder of "main.py," and then just run it. </br>
The model would be saved in the "save_dir" folder set in config.ini </br>

### prediction
After put the inputs one wants to predict into the folder: "pred_inputs," </br>
just run "pred.py" and the results will be saved in the folder: "pred_results" </br>
Notice that the paths of inputs and outputs of predictions are also defined in "config.ini"

## Showcase
Prediction on the date in 2019.03.25: </br>
![image](https://github.com/AugChiang/Climate-downscaling/blob/main/Example_20190325.png)

## Environment
python == 3.10.3 </br>
tensorflow == 2.12.0 </br>
(tested on 2023.07.05)
