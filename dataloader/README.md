This module preprocess downloaded medical dataset into new one usefull from ML perspecitves, with 
images and segmntations saved numpy arrays, and necessary metadata saved in json files.

Given dataset, with manifest file (made with data_handler module), and the collection_name
script make_numpy_arrays.py computes paths to all scans and finds coresponding rtstructs.
With that pairing dictionary, it loads the scan and computes numpy array out of it and saves inte the same
folder as dicom. Then it load coresponding rtstructs, and makes numpy array with correct orientation (the same as scan). 
Then segmentation masks is saved in the same folder as rtstruct file.

The file also checks if the numpy arrays for given pair are already made. If so, it skips them.

Script make_color_dicts.py is similar to the make_numpy_arrays.py
Given dataset, with manifest file (made with data_handler module), and the collection_name
it computes and saves the segmentation_labels.json with the label names and coresponding colors for visualization. 

