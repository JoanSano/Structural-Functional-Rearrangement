import logging
import shutil
import os

from utils.paths import *
from utils.steps import *

def anatomical_steps(f, output_directory, whole_path, subjectID, session, mri, name, acq, weight, extension, config):
    """
    Performs the preprocessing steps in anatomical images.
    """
    output_directory = output_directory + 'co-registered/'
    logging.info(f" PreProcessing {subjectID} in session {session} with mri {mri}")
    steps = []

    # Relevant paths
    file_dir = output_directory+subjectID+'/'+session+'/'
    mri_path = bids_tree(file_dir, mris=config["subjects"]["mris"])[mri]
    int_path, at_path = mri_path+'intermediate/', file_dir+'atlas/'
    check_path(int_path)
    if config["reference"] != "atlas":
        check_path(at_path)

    # Brain Extraction
    int_output, mask = brain_extraction(input=f, output=int_path+name+'_bet', skip=config["data"]["skip"])
    steps.append("Brain Extraction")
    logging.info(f" {subjectID} Brain extraction done")

    # Bias field correction
    int_output = bias_correction(input=int_output, output=int_path+name+'_biasC', skip=config["data"]["skip"], mask=mask)
    steps.append("Bias Correction")
    logging.info(f" {subjectID} Bias field correction done")

    # Registration // creation of reference+upsample
    if config["reference"] == 'dwi':
        # Reference is DWI
        reference_img, _ = get_folders(search_type='file', path=at_path, subjects='ref.nii')
        
        # A sanity check of whether the file exists or not to throw an error and report it in logs.txt
        try:
            with open(reference_img[0], 'r') as check_file: 
                pass
        except:
            with open('reference_image', 'r') as check_file: 
                pass
        
        # Register to the reference image
        int_output, tr_matrix = register_to_reference(
            input=int_output, output=int_path+name+'_reg', reference=reference_img[0],skip=config["data"]["skip"]
        )
        reg_mask, _ = register_to_reference(
            input=mask, output=int_path+name+'_reg_mask', reference=reference_img[0], skip=config["data"]["skip"], matrix=tr_matrix
        )
        _, atlas_matrix = register_to_reference(
            input=config["atlas"]["REF"], output=at_path+name+'_atlas', reference=int_output, skip=config["data"]["skip"]
        )
        register_to_reference(
            input=config["atlas"]["labels"], output=at_path+name+'_labels', reference=int_output, skip=config["data"]["skip"], matrix=atlas_matrix
        )
        steps.append("Anat&Labels Registered_to_dwi")

        # Copy the transformation matrix to the output directory
        shutil.copyfile(tr_matrix, mri_path+name+'_transform.txt') 

        # Upsample to anat resolution
        int_output = upsample(input=int_output, output=mri_path+name, skip=config["data"]["skip"], factor=config["vox_size"]["anat"])
        upsample(input=reg_mask, output=mri_path+name+'_mask', skip=config["data"]["skip"], factor=config["vox_size"]["anat"])
        steps.append("Upsampled to %s mm" % config["vox_size"]["anat"])
    elif config["reference"] == 'anat':
        # Reference is Anat
        int_output = upsample(int_output, output=mri_path+name, skip=config["data"]["skip"], factor=config["vox_size"]["anat"])
        upsample(input=mask, output=mri_path+name+'_mask', skip=config["data"]["skip"], factor=config["vox_size"]["anat"])
        steps.append("Upsampled to %s" % config["vox_size"]["anat"])

        _, atlas_matrix = register_to_reference(
            input=config["atlas"]["REF"], output=at_path+name+'_atlas', reference=int_output, skip=config["data"]["skip"]
        )
        register_to_reference(
            input=config["atlas"]["labels"], output=at_path+name+'_labels', reference=int_output, skip=config["data"]["skip"], matrix=atlas_matrix
        )
        steps.append("Atlas Registered_to_anat")
    elif config["reference"] == 'atlas':
        # Reference is Atlas
        reference_img = config["atlas"]["REF"]
        int_output, tr_matrix = register_to_reference(
            input=int_output, output=mri_path+name, reference=reference_img, skip=config["data"]["skip"]
        )
        reg_mask, _ = register_to_reference(
            input=mask, output=mri_path+name+'_mask', reference=reference_img, skip=config["data"]["skip"], matrix=tr_matrix
        )
        steps.append("Registered to Atlas")   

        # Upsample to anat resolution
        int_output = upsample(input=int_output, output=mri_path+name, skip=config["data"]["skip"], factor=config["vox_size"]["anat"])
        upsample(input=reg_mask, output=mri_path+name+'_mask', skip=config["data"]["skip"], factor=config["vox_size"]["anat"])
        steps.append("Upsampled to %s mm" % config["vox_size"]["anat"]) 
    else:
        raise ValueError("Reference image not recognized")
    logging.info(f" {subjectID} Registration steps done")

    # Copy json to derivatives
    shutil.copyfile(whole_path+name+'.json',mri_path+name+'.json')

    # Keep intermediate
    if not config["data"]["keep_intermediate"]:
        os.system(f"rm -rf {int_path}")

    ### Output summary ###
    logging.info(f" Subject {subjectID} in session {session} with mri {mri} \n with completed preprocessing steps: {', '.join(steps)}")

if __name__ == '__main__':
    pass