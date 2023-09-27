# Train a Mask-RCNN model on Encord Annotate and Encord Active Projects

This project can be used to train a MaskRCNN model on Encord Annotate and Active projects. 
The trained model, then, can be used to generate predictions.

First, setup conda the environment by using the following command:

```shell
conda env create -f environment.yml 
```

Once the conda virtual env is set up, activate the environment:

```shell
conda activate encord-maskrcnn 
```

## 1. Encord Annotate

### 1.1 Download Encord project data (images and videos)
1. Create `config.ini`. and add a section called `[ENCORD]`. 
Fill the properties following the `example_config.ini` file
2. Inside the project folder run 
```shell
python download_encord_data.py
```
3. It will download all images and (single images and image groups) video frames
to the specified folder.



### 1.2. Export annotations in COCO format
1. Got to the project in Encord platform, click Export tab.
2. Choose COCO in Export options, and select the data units that you want
to include labels of on the right side. Click Export. When the exporting process
is finished, the bell shape on the upper right corner will indicate that.
3. Download the exported COCO file and store it in your machine where the 
training will take place.

### 1.3 Training model
1. Open the `config.ini` file and fill the sections in `[DATA]`, `[LOGGING]`, 
and `[TRAIN]` sections.
2. 