There is file main that allows to browse given dataset in NBIA, and downloads specified
studies. May filter by modalities. Extend this to select individual studies.

Data is downloaded to specified path, in the folder named fater collection name.
Each series or its segmentation is saved in subfolder with study index and under another subfolders 
with the series UID.

There is pipeline that based on the downloaded data makes JSON file named manifest, that
contains information sufficient to recover all data (make api calls for those UIDs and 
redownload them)

New folder with manifest file is sufficient to recover dataset.


There is a function x_y_pairing that loads the manifest and scraps information in series,
and returns pairs: image: segmentation.

Then, with that pairs we can make the 3D numpy arrays.


TODO
Make more general functions, that are converting whole series to the numpy - work
on individual folder, not whole pairing at once.

