import os
import logging

from utils.paths import get_folders, get_info_from_nifti, proceed, check_path
from dwi_preprocess import dwi_steps
from anatomical_preprocess import anatomical_steps
from fmri_preprocess import fmri_steps
from utils.steps import functional_coregistration

def preprocess_subject(subject, output_directory, config):
    """
    Perform FSl and MRtrix3 preprocessing steps for all sessions and mris specified in the config file.
    """

    ### Get files from the subject ###
    exclude = '*backup*' if config["data"]["backup"] else ''
    files, Nfiles = get_folders(search_type='file', path=subject, maxdepth=3, exclude=exclude, subjects='.gz' if config["data"]["gzip"] else '.nii')
    
    ### Preprocess files ###
    for f in files: 
        whole_path, subjectID, session, mri, name, acq, weight, extension = get_info_from_nifti(f)
 
        if proceed(session, mri, acq, config):
            ##############################################################
            ### Conditions are met for starting with the preprocessing ###
            ##############################################################
            check_path(output_directory)
            try:
                if mri == 'anat':
                    # Get rid of .gz extension
                    f = '.'.join(f.split('.')[:-1]) if config["data"]["gzip"] else f
                    anatomical_steps(f, output_directory, whole_path, subjectID, session, mri, name, acq, weight, extension, config)
                elif mri == 'dwi':
                    # Get rid of .gz extension
                    f = '.'.join(f.split('.')[:-1]) if config["data"]["gzip"] else f
                    dwi_steps(f, output_directory, whole_path, subjectID, session, mri, name, acq, weight, extension, config)
                elif mri == 'func': 
                    # fmriprep does not allow to preprocess one session alone, so we only preprocess one of the two
                    # The summary that xcp-d outputs controls whether a subject has already been preprocessed
                    if not os.path.exists(output_directory+'xcp_d/'+subjectID+'.html'):
                        fmri_steps(output_directory, config, subjectID, img_tag=config["fmri"]["image_tag"])
                    else:
                        logging.info(f" {subjectID} in {session}: BOLD preprocessed available ... skipping")
                    # Co-register the tumor to BOLD space (Documentation for details)
                    logging.info(f" {subjectID} co-registering BOLD {session} to MNI-anatomical space")
                    functional_coregistration(output_directory, subjectID, session, config["fmri"]["NS_REG"])

            except Exception as e:
                logging.error(f" Error in preprocessing {subjectID} - {session} - {mri}")
                if config["data"]["logs"]:
                    check_path(output_directory+'metadata/')
                    with open(output_directory+'metadata/logs.txt', 'a') as logs_file:
                        logs_file.write(f"Failing to preprocess {name} \n")
                        logs_file.write(f"{subjectID} in {session} and mri {mri} with {acq} - {weight} \n\n")
                        logs_file.write(f"Full name of the file:\n{f} \n\n")
                        logs_file.write(f"Error: {e} \n")
                        logs_file.write("============================================================================================\n")
        os.system(f"gzip {whole_path+name}.nii")

if __name__ == '__main__':
    pass