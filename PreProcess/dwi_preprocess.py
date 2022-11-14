import logging
import shutil
import os

from utils.paths import *
from utils.steps import *
from utils.acq_index_files import find_secondary_acquisitions

def dwi_steps(f, output_directory, whole_path, subjectID, session, mri, name, acq, weight, extension, config):
    """
    Performs the preprocessing steps in diffusion weighted images.
    """
    output_directory = output_directory + 'co-registered/'
    logging.info(f" PreProcessing {subjectID} in session {session} with mri {mri}")
    steps = []

    # Relevant paths
    file_dir = output_directory+subjectID+'/'+session+'/'
    mri_path = bids_tree(file_dir, mris=config["subjects"]["mris"])[mri]
    int_path, at_path = mri_path+'intermediate/', file_dir+'atlas/'
    sec_path = mri_path+'intermediate/secondary/'
    check_path(int_path)
    check_path(sec_path)

    ### Topup Susceptiility Artifacts Estimation ###
    sec_files, available_secs = find_secondary_acquisitions(whole_path, subjectID, session, acq, weight)
    if available_secs:
        # Prepare data for topup and eddy
        b0_vols, topup_data = prepare_topup(
            input=f, tmp_dir=int_path, raw_dir=whole_path, sec_files=sec_files, skip=config["data"]["skip"], vols=1
        )
        int_output, eddy_acq, eddy_index, bvecs, bvals = prepare_eddy(
            input=f, tmp_dir=int_path, raw_dir=whole_path, sec_files=sec_files, merge=config["merge_acqs"]
            )

        # TopUp can be performed only if there are at least one secondary acquisitions
        unwarped, unwarped_mask = topup(input=b0_vols, output=sec_path+'topup', datain=topup_data, skip=config["data"]["skip"])
        steps.append("Top Up off-resonance field map")

        logging.info(f" {subjectID} Suceptibility artifacts done")
    else:
        # No susceptibility - then prepare for eddy
        int_output, eddy_acq, eddy_index, bvecs, bvals = prepare_eddy(input=f, tmp_dir=int_path, raw_dir=whole_path)

    # Brain Extraction 
    int_output, mask = brain_extraction(input=int_output, output=int_path+name+'_bet', skip=config["data"]["skip"], method=config["dwi_be"])
    steps.append("Brain Extraction")
    logging.info(f" {subjectID} Brain extraction done")

    # Denoise
    int_output = MP_PCA_denoise(input=int_output, output=int_path+name+'_mppca', skip=config["data"]["skip"], mask=mask)
    steps.append("MP_PCA_denoise")
    logging.info(f" {subjectID} Denoising done")

    # Gibbs artifacts
    int_output = gibbs_unringing(input=int_output, output=int_path+name+'_gibbs', skip=config["data"]["skip"])
    steps.append("Gibbs_unringing")
    logging.info(f" {subjectID} Gibbs artifacts removal done")

    # Eddy + Motion + Bmatrix Correction 
    tpd = sec_path+'topup' if available_secs else None
    ed_mask = unwarped_mask if available_secs else mask
    int_output, rotated_bvecs = eddy(
        input=int_output, output=int_path+name+'_eddy', directory=sec_path, acq=eddy_acq, index=eddy_index,
        bvecs=bvecs, bvals=bvals, mask=ed_mask, skip=config["data"]["skip"], topup_data=tpd, cuda=config["eddy"]["cuda"], version=config["eddy"]["version"]
    )
    steps.append("Eddy-Motion-B correction")
    logging.info(f" {subjectID} Eddy, Motion and Bmatrix correction done")

    # Remove eddy added background 
    int_output, mask = brain_extraction(input=int_output, output=int_path+name+'_eddy_bet', skip=config["data"]["skip"], method=config["dwi_be"])
    steps.append("Brain Extraction")
    logging.info(f" {subjectID} Eddy background removal done")

    # Registration // creation of reference+upsample
    if config["reference"] == 'dwi':
        if config["reference"] != "atlas":
            check_path(at_path)
        
        # Copy bval and rename bvec to main derivatives folder 
        shutil.copyfile(rotated_bvecs, mri_path+name+'.bvec')
        shutil.copyfile(whole_path+name+'.bval',mri_path+name+'.bval')

        # Reference is DWI
        int_output = upsample(input=int_output, output=mri_path+name, skip=config["data"]["skip"], factor=config["vox_size"]["dwi"])
        upsample(input=mask, output=mri_path+name+'_mask', skip=config["data"]["skip"], factor=config["vox_size"]["dwi"])
        steps.append("Upsampled to %s mm" % config["vox_size"]["dwi"])

        reference_img = create_reference(int_output, output=at_path+name+'_ref', skip=config["data"]["skip"])
        steps.append("Created 3D DWI reference image")
    elif config["reference"] == 'anat':
        # Reference is Anat
        reference_img = file_dir+'anat/'+subjectID+'_'+session+'_'+'T1w.nii.gz'
        int_output, tr_matrix = register_to_reference(
            input=int_output, output=int_path+name+'_reg', reference=reference_img, skip=config["data"]["skip"], dim=4
        )
        reg_mask, _ = register_to_reference(
            input=mask, output=int_path+name+'_reg_mask', reference=reference_img, skip=config["data"]["skip"], matrix=tr_matrix
        )
        steps.append("Registered to Anat")

        # B-Matrix rotation
        rotate_Bmatrix(bvecs=rotated_bvecs, bvals=bvals, output=mri_path+name, matrix=tr_matrix)
        steps.append("BMatrix Reorientation")

        # Copy the transformation matrix to the output directory
        shutil.copyfile(tr_matrix, mri_path+name+'_transform.txt') 

        # Upsample to dwi resolution
        int_output = upsample(input=int_output, output=mri_path+name, skip=config["data"]["skip"], factor=config["vox_size"]["dwi"])
        upsample(input=reg_mask, output=mri_path+name+'_mask', skip=config["data"]["skip"], factor=config["vox_size"]["dwi"])
        steps.append("Upsampled to %s mm" % config["vox_size"]["dwi"])
    elif config["reference"] == 'atlas':
        # TODO: Need further testing, for some reason the output is filled with zeros!

        # Reference is Atlas
        reference_img = config["atlas"]["REF"]
        int_output, tr_matrix = register_to_reference(
            input=int_output, output=mri_path+name, reference=reference_img, skip=config["data"]["skip"], dim=4
        )
        reg_mask, _ = register_to_reference(
            input=mask, output=mri_path+name+'_mask', reference=reference_img, skip=config["data"]["skip"], matrix=tr_matrix
        )
        steps.append("Registered to Atlas")

        # B-Matrix rotation
        rotate_Bmatrix(bvecs=rotated_bvecs, bvals=bvals, output=mri_path+name, matrix=tr_matrix)
        steps.append("BMatrix Reorientation")
        
        # Copy the transformation matrix to the output directory
        shutil.copyfile(tr_matrix, mri_path+name+'_transform.txt') 

        # Upsample to dwi resolution
        int_output = upsample(int_output, output=mri_path+name, skip=False, factor=config["vox_size"]["dwi"])
        upsample(input=reg_mask, output=mri_path+name+'_mask', skip=False, factor=config["vox_size"]["dwi"])
        steps.append("Upsampled to %s mm" % config["vox_size"]["dwi"])
    else:
        raise ValueError("Reference image not recognized")
    logging.info(f" {subjectID} Registration steps done")


    # Delete concatenated from original
    os.remove(bvecs)
    os.remove(bvals)

    # Copy json to derivatives
    shutil.copyfile(whole_path+name+'.json',mri_path+name+'.json')

    # Keep intermediate
    if not config["data"]["keep_intermediate"]:
        os.system(f"rm -rf {int_path}")

    ### Output summary ###
    logging.info(f" Subject {subjectID} in session {session} with mri {mri} \n with completed preprocessing steps: {', '.join(steps)}")

if __name__ == "__main__":
    pass