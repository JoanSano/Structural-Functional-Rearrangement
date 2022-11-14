import os
from subprocess import Popen, STDOUT, PIPE
import logging
import re

def check_path(path):
    """
    Check if specified path exist. Make directory in case it doesn't.
    """
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
    return path

def get_subjects(config):
    """
    Get all the subjects specified in the config file.
    """

    dataset_path = config['paths']['dataset_dir'] + 'derivatives/'
    subject_ID = config['paths']['subject']
    session = config['paths']['session']

    logging.info(" Target Dataset: " + dataset_path)
    output_dwi = Popen(f"find {dataset_path if len(dataset_path)>0 else '.'} ! -path '*/intermediate/*' -wholename \'*/{subject_ID}/{session}/dwi/*_dwi.nii*\'",
                                        shell=True, stdout=PIPE)
    files_dwi = str(output_dwi.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    logging.info(" Found " + str(len(files_dwi)) + " subject(s) to process")

    return files_dwi

def get_files(f, lesion):
    """
    Get all the relevant files for the reconstruction of the connectome.
    """

    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    dwi_path = f.removesuffix(f.split('/')[-1])
    anat_path = f.split('/dwi/')[0] + '/anat/'
    name_dwi = (f.split('/')[-1]).split('.')[0] # just the file name (without extension)
    subject_ID = str(re.findall("sub-...[0-9]+", name_dwi)[0]) 
    session = name_dwi.split("_")[1] #str(re.findall("_ses-p.+p", name_dwi)[0]) 
    #acq = str(re.findall("_acq-.{2}", name_dwi)[0])

    logging.info(f" Processing " + subject_ID + " in session " + session.split('-')[-1])

    nii_dwi = dwi_path + name_dwi + '.nii.gz'
    bval_dwi = dwi_path + name_dwi + '.bval'
    bvec_dwi = dwi_path + name_dwi + '.bvec'
    json_dwi = dwi_path + name_dwi + '.json'
    mask_dwi = dwi_path + name_dwi + '_mask.nii.gz'
    nii_t1 = anat_path + subject_ID + '_' + session + '_T1w.nii.gz'
    mask_t1 = anat_path + subject_ID + '_' + session + '_T1w_mask.nii.gz'
    tumor_t1 = anat_path + subject_ID + '_' + session + '_T1w_' + lesion +'.nii.gz'

    return subject_ID, session, nii_dwi, mask_dwi, bval_dwi, bvec_dwi, json_dwi, nii_t1, mask_t1, tumor_t1

def output_directory(f, dataset_path, acronym, custom_output=None):
    """
    Create and check the output directory for the current subject processed in f.
    """

    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    path = f.removesuffix(f.split('/')[-1])
    
    if custom_output is not None:
        output_directory = custom_output + acronym + re.split('derivatives/', re.split(dataset_path, path)[-1])[-1]
    else:
        output_directory = dataset_path + acronym + re.split('derivatives/', re.split(dataset_path, path)[-1])[-1]
    output_directory = re.split('/dwi', output_directory)[0] + '/'
    intermediate_dir = output_directory + 'MRtrix3Files/'

    check_path(output_directory)
    check_path(intermediate_dir)

    return output_directory, intermediate_dir