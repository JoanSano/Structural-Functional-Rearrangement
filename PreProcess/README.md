# PreProcess
Perform precprocessing on dwi and t1 images. Easy extension to fMRI.

Dependencies to install: FSL, MRtrix3, ANTs and DIPY.

The idea is to modify (or create your own) config.yaml file and add it the preprocess.sh script. 
This file should be pretty self-explanatory.

The steps performed depend on the dataset and should automatically adapt to the available data.
For example, topup will be performed if and only if a REVERSED encoding file is present. The index
and aquisition files for eddy are written (automatically) depending on what is set in the json. You 
should be careful enough to tell the config file which are the possible main phase encoding directions
to be found in the dataset. It is also possible to merge all this acquisitions to a single file and 
pass it to eddy.

The brain extraction step is the one you should be careful. The force (-f) and gradient (-g) parameters
of FSL's bet2 should be inspected before running the script on the whole dataset to avoid wrong 
brain extractions. This can be found in the utils/steps.py in the brain_extraction function. It is 
also possible to run bet with DIPY's median_otsu for the diffusion data, but be sure that it works well.
