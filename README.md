<h1>Instructions</h1>

Raw data should be in folder **data**. The script _create_data.py_ will process them into Bird eye view format for segmentation specified in config file and store them in **dataset/processed**. These data can be moved into training(**trn**)/validation(**val**)/testing(**tst**) folder in **dataset**. _The script create_data.py is set to one sequence only for example. Loop for processing can be set in the ending lines_.

Implemented segmentation models are SqueezeSegv2(2018) and Unet (2015). They are kind a old, so probably not desired.

One sequence of data has 3 files (CLCA, PSVH, 20...). CLCA has lidar scans, PSVH is the output from the Valeo primary system detecting road boundaries and cars. 20* are human annotations in form of bounding box for cars a and connecting points for road boundary (Best understanding can be acquired from function in code _create_data.py_.
