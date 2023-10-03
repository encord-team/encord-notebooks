# Train a Mask-RCNN model on Encord Annotate Projects

This project can be used to train a MaskRCNN model on Encord Annotate projects. 
The trained model, then, can be used to generate predictions for the Encord Active.

First, setup conda the environment by using the following command:

```shell
conda env create -f environment.yml 
```

Once the conda virtual env is set up, activate the environment:

```shell
conda activate encord-maskrcnn 
```

## 1. Download Encord project data (images and videos)
1. Create `config.ini`. and add a section called `[ENCORD]`. 
Fill the properties following the `example_config.ini` file
2. Inside the project folder run 
```shell
python download_encord_data.py
```
3. It will download all images (single images and image groups) and video frames
to the specified folder.



## 2. Export annotations in COCO format
1. Go to the project in Encord platform, click Export tab.
2. Choose COCO in Export options, and select the data units that you want
to include labels of on the right side. Click Export. When the exporting process
is finished, the bell shape on the upper right corner will indicate that.
3. Download the exported COCO file and store it in your machine where the 
training will take place.

## 3. Training model
1. Open the `config.ini` file and fill the sections in `[DATA]`, `[LOGGING]`, 
and `[TRAIN]` sections following the `example_config.ini` file.
2. Run the following command in the project folder:

```shell
python train.py
```
 3. During the training best checkpoint and the last checkpoint will be saved to the experiment wandb 
folder if you enabled the wandb logging, if not, it will be saved directly in the project folder.

## 4. Inference on Encord Active project
1. Make sure that Encord Active project has data downloaded in `local-data` folder. If not, import the 
project by enabling storing data locally.
2. Open the `config.ini` file and fill the `[INFERENCE]` section.
3. run `python inference_on_active_project.py`
4. A prediction .pkl file will be generated next to checkpoint file. Use this file to import predictions 
to the Encord Active by using `encord-active import predictions predictions_123.pkl -t /path/to/encord-active -db` 
command.
5. Once the importing is finished, refresh the browser running the Encord Active.