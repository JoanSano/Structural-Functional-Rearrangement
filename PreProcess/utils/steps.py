import os
from subprocess import Popen, STDOUT, PIPE
from pathlib import Path
import shutil

from utils.acq_index_files import acq_file, index_file
from utils.paths import check_path

from dipy.segment.mask import median_otsu 
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, reorient_bvecs
from nibabel import load, Nifti1Image, save
import numpy as np

def brain_extraction(input, output, skip, method=None):
    """
    Skull stripping using FSL's bet2 (for anatomical) and DIPY's median otsu (for diffusion).
    """
    mask = output + '_mask.nii.gz'
    if skip and os.path.exists(mask) and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz', mask
    else: 
        if 'T1w' in input:
            os.system(
                f"bet2 {input} {output} -f 0.5 -g -.6"
            )
            os.system(
                f"bet2 {output} {output} -f 0.35 -m"
            )
        else:
            if method == "median_otsu":
                img = load(input)
                data = img.get_fdata()
                vol_idx = range(len(data[0,0,0,:])) if len(data.shape)==4 else None
                bet_data, mask_data = median_otsu(data, vol_idx=vol_idx, numpass=5, median_radius=5, dilate=3, autocrop=False) 
                # Cropping the mask introduces dimensionality issues in eddy when using the unwarped masked b0 volume
                save(Nifti1Image(bet_data, img.affine, img.header), output+'.nii.gz')
                save(Nifti1Image(mask_data, img.affine, img.header), mask)
            elif method == "bet2":
                os.system( # Extract the first volume
                    f"fslroi {input} {output}_b0volume 0 1"
                )
                os.system( # Mask the first volume
                    f"bet2 {output}_b0volume {output} -f 0.19 -m"
                )
                os.system( # Apply the mask to all the volumes
                    f"fslmaths {input} -mas {mask} {output}"
                )
            elif method is None:
                print("--- No brain extraction method is provided ---")
            else:
                raise ValueError("Brain Extraction method not implemented")
        return output+'.nii.gz', mask

def MP_PCA_denoise(input, output, mask, skip):
    """
    Marchenko-Pasteur PCA denoising using MRtrix3.
    """
    if skip and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz' 
    else:
        os.system(
            f"dwidenoise {input} {output}'.nii.gz' -mask {mask} -force -quiet"
        )
        return output+'.nii.gz' 

def gibbs_unringing(input, output, skip):
    """
    Removal of Gibbs ringing artifacts using MRtrix3.
    """
    if skip and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz' 
    else:
        os.system(
            f"mrdegibbs {input} {output}'.nii.gz' -force -quiet"
        )
        return output+'.nii.gz' 

def register_to_reference(input, output, skip, reference, matrix=None, dim=3):
    """
    #TODO: Add description
    """
    if skip and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz', output + '_transform.txt'
    else:        
        if dim == 3:
            if matrix is None: # Find the linear transformation
                transform_mat = output + '_transform.txt'
                os.system(
                    f"flirt -in {input} -ref {reference} -out {output} -omat {transform_mat}"
                )
                return output+'.nii.gz', transform_mat
            else: # Apply the transformation
                os.system(
                    f" flirt -in {input} -ref {reference} -out {output} -applyxfm -init {matrix}"
                )
                return output+'.nii.gz', matrix
        elif dim == 4:
            if matrix is None: # Find the linear transformation
                # It is assumed that the 4D image has been corrected for motion artifacts so all slices are coregistered!
                tempdir = 'tmp-reg_' + output.split('/')[-1] + '/' #+output.split('/')[-4]+'/'
                check_path(tempdir)
                # Split allthe volumes into separate 3D files
                os.system(
                    f"fslsplit {input} {tempdir} -t"
                )
                transform_mat = output + '_transform.txt'
                # Find the transformation with the first volume
                os.system(
                        f"flirt -in {tempdir}'0000.nii.gz' -ref {reference} -out {tempdir}'0000.nii.gz' -omat {transform_mat}"
                    )
                # Apply the transformation to the rest of the volumes
                for vol in os.listdir(tempdir):
                    if vol != '0000.nii.gz':
                        os.system(
                            f" flirt -in {tempdir+vol} -ref {reference} -out {tempdir+vol} -applyxfm -init {transform_mat}"
                        )
                # Merge all the volumes
                os.system(
                    f"fslmerge -t {output} {tempdir}*.nii.gz"
                )
                # Remove the temporary directory
                os.system(
                    f"rm -rf {tempdir}"
                )
                return output+'.nii.gz', transform_mat
            else: #Apply the transformation
                os.system(
                    f" flirt -in {input} -ref {reference} -out {output} -applyxfm -init {matrix}"
                )
                return output+'.nii.gz', matrix
        else:
            raise ValueError('Dimension must be either 3 or 4')

def bias_correction(input, output, skip, mask):
    """
    Bias correction using N4 ants Algorithm.
    """
    if skip and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz'
    else:
        os.system( 
            f"N4BiasFieldCorrection -i {input} -o {output}'.nii.gz' -d 3 -x {mask} -r"
        )
        return output+'.nii.gz'

def upsample(input, output, skip, factor=1.5):
    """
    MRtrix3 upsampmling image to increase/decrease resolution.
    """
    if skip and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz'
    else:
        os.system( 
            f"mrgrid {input} regrid {output}'.nii.gz' -force -quiet -voxel {factor}"
        )
        return output+'.nii.gz'

def create_reference(input, output, skip):
    """
    Create a template image for registration. Useful for registering to dwis, since anatomical
        and atlases are already 3D images.
    """
    if skip and os.path.exists(output+'.nii.gz'):
        return output+'.nii.gz'
    else:
        os.system(
             f"fslroi {input} {output} 0 1"
            )
        return output+'.nii.gz'
def extract_b0s(inputs, output, vols=None):
    """
    Extract the b0 volumes from a DWI image.
    Returns the number of b0 volumes extracted.
    Saves the b0 volumes in the directory specified.
    Options: Extract the specified number of b0 values if vols is not None
    """
    # Extract all b0 volumes
    os.system(
        f"dwiextract {inputs[0]} {output} -bzero -fslgrad {inputs[2]} {inputs[1]} -force -quiet"
    )
    # Keep the specified number of b0 volumes
    if vols is not None:
        os.system(
            f"mrconvert {output} {output} -coord 3 0:{vols-1} -force -quiet"
        )
    # Report the final number of volumes extracted
    message = Popen(f"fslnvols {output}", shell=True, stdout=PIPE).stdout.read()
    Nvolumes = int(str(message).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')[0])
    return output, Nvolumes

def prepare_topup(input, tmp_dir, sec_files, raw_dir, skip, vols=None):
    """
    Prepare the images and files to be used in the topup step.
    ###
    # TODO: Improve what is said below
    Here it is assumed that all the acquisitions are co-registered.
        Might not be exactly true, but for these datasets it approximately holds.
        The fact that this is not exactly true will cause a WARNING whehn merging the volumes
        from different acquisitions.
    ####    
    """
    output = tmp_dir+input.split('.')[0].split('/')[-1]+'_b0vols'
    if skip and os.path.exists(output+'.nii.gz') and os.path.exists(tmp_dir+'topup_data.txt'):
        return output+'.nii.gz', tmp_dir+'topup_data.txt'
    else:
        # List of main files in the scheme
        main_file = input.split('.')[0]  
        main_files = [main_file+'.nii.gz', main_file+'.bval', main_file+'.bvec', main_file+'.json']
        ### These two lines are useful if and only if a bet has been run at the very begining of the pipeline
        #origin_name = raw_dir+'_'.join(input.split('.')[0].split('/')[-1].split('_')[:-1])
        #main_files = [main_file+'.nii.gz', origin_name+'.bval', origin_name+'.bvec', origin_name+'.json']

        # Extract b0 volumes from main file
        jsons, suffix = [], '_b0s'
        _, Nvs = extract_b0s(main_files, tmp_dir+'main'+suffix+'.nii.gz', vols=vols)
        for i in range(Nvs):
            jsons.append(main_files[3]) # append the jsons as many times a b0s

        # Extract b0 volumes from secondary files
        for i in range(len(sec_files)):
            _, Nvs = extract_b0s(sec_files[i], tmp_dir+'sec'+str(i)+suffix+'.nii.gz', vols=vols)
            for j in range(Nvs):
                jsons.append(sec_files[i][3]) # append the jsons as many times a b0s

        # Merge the b0 files
        os.system(
            f"fslmerge -t {output} {tmp_dir}*{suffix}.nii.gz"
        )
        os.system(
            f"rm {tmp_dir}*{suffix}.nii.gz"
        )
        # Create the acqparams file for the --datain option in topu up
        acq_file(jsons, tmp_dir+'topup_data.txt')
        return output+'.nii.gz', tmp_dir+'topup_data.txt'

        
def topup(input, output, datain, skip):
    """
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/TopupUsersGuide
    Runs topup fsl command to estimate the off-ressonance field.
    Additionally it performs a brain extraction on the unwarped b0 volumes to output a mask for eddy.
    This function needs prepare_topup to be run first to run topup on the bo volumes from all the acquisitions.
    Returns the unwarped b0 volumes and the mask estimated from them to pass as input to eddy (suggested in FSL wiki).
    """
    if skip and os.path.exists(output+'_movpar.txt') and os.path.exists(output+'_fieldcoef.nii.gz') and os.path.exists(output+'_unwarped.nii.gz'):
        return output+'_unwarped.nii.gz', output+'_unwarped_mask.nii.gz'
    else:
        os.system(
            f"topup --imain={input} --datain={datain} --out={output} --iout={output}_unwarped"
        )
        os.system(
            f"fslmaths {output}_unwarped -Tmean {output}_unwarped"
        )
        b0s_unwarped, b0s_unwarped_mask = brain_extraction(input=output+'_unwarped.nii.gz', output=output+'_unwarped', skip=False, method="bet2")
        os.system(
            f"rm {'/'.join(datain.split('/')[:-1])}/*.topup_log"
        )
        return b0s_unwarped, b0s_unwarped_mask

def concatenate_gradients(bvals, bvecs, output):
    """
    Concatenate bvals and bvecs from different acquisitions into a single file.
    """
    cat_bvals = '_'.join(output.split('_')[:-1])+'_concat.bval'
    cat_bvecs = '_'.join(output.split('_')[:-1])+'_concat.bvec'

    # Read and concatenate bvals and bvecs
    bval_lines, bvec_lines = '', ['', '', '']
    for i, j in zip(bvals, bvecs):
        with open(i, 'r') as bval_table:
            current = bval_table.readlines()[0].strip()
            bval_lines = bval_lines + current + ' ' 
        with open(j, 'r') as bvec_table:
            current = bvec_table.readlines()
            bvec_lines[0] = bvec_lines[0] + current[0].strip() + ' '
            bvec_lines[1] = bvec_lines[1] + current[1].strip() + ' ' 
            bvec_lines[2] = bvec_lines[2] + current[2].strip() + ' ' 
    
    # Write the concatenated bvals and bvecs
    with open(cat_bvals, 'w') as bval_table:
        bval_table.write(bval_lines)
    with open(cat_bvecs, 'w') as bvec_table:
        bvec_table.write('\n'.join(bvec_lines))
    return cat_bvecs, cat_bvals

def prepare_eddy(input, tmp_dir, raw_dir, sec_files=None, merge=None):
    """
    Prepares the images and files to be used in the eddy step.
    Based on the configuration and the secondary acquisitions,
        it will merge everythin sequentially and the index file
        will be created accordingly. Feel free to merge with other schemes.

    ###
    # TODO: Improve what is said below
    Here it is assumed that all the acquisitions are co-registered.
        Might not be exactly true, but for these datasets it approximately holds.
        The fact that this is not exactly true will cause a WARNING whehn merging the volumes
        from different acquisitions.
    ####
    """
    eddy_acq = tmp_dir+'eddy_acq.txt'
    eddy_index = tmp_dir+'eddy_ind.txt'

    # List of main files in the scheme
    main_file = input.split('.')[0]    
    files = [[main_file+'.nii.gz', main_file+'.bval', main_file+'.bvec', main_file+'.json']]
    ### These two lines are useful if and only if a bet has been run at the very begining of the pipeline
    #origin_name = raw_dir+'_'.join(input.split('.')[0].split('/')[-1].split('_')[:-1])
    #files = [[main_file+'.nii.gz', origin_name+'.bval', origin_name+'.bvec', origin_name+'.json']]

    # Merge files if specified
    if merge and sec_files is not None:
        merged_file = tmp_dir+'merged_acquisitions.nii.gz'
        os.system(f"cp {input} {merged_file}")
        for i in range(len(sec_files)):
            files.append(sec_files[i])
            os.system(
                f"fslmerge -t {merged_file} {merged_file} {sec_files[i][0]}"
            )
    else:
        merged_file = input

    # Create the acquisition and index parameters file
    jsons, niis = [files[i][-1] for i in range(len(files))], [files[i][0] for i in range(len(files))]
    bvals, bvecs = [files[i][1] for i in range(len(files))], [files[i][2] for i in range(len(files))]
    acq_file(jsons, eddy_acq)
    index_file(niis, eddy_index)
    cat_bvecs, cat_bvals = concatenate_gradients(bvals, bvecs, main_file)
    return merged_file, eddy_acq, eddy_index, cat_bvecs, cat_bvals


def eddy(input, output, directory, acq, index, bvecs, bvals, mask, skip, topup_data=None, cuda=False, version=None):
    """
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide
    Runs eddy, motion and b-matrix correction according to previous steps (prepare_topup, topup, prepare_eddy).
    cuda: controls which version of eddy is used
    """
    basename = directory + 'eddy_output'
    if skip and os.path.exists(output+'.nii.gz') and os.path.exists(output+'.bvec'):
        return output+'.nii.gz', output+'.bvec'
    else:
        ed_type = 'cuda'+str(version) if cuda else 'openmp'
        if topup_data is None:
            os.system(
                f"eddy_{ed_type} --imain={input} --mask={mask} --acqp={acq} --index={index} --bvecs={bvecs} --bvals={bvals} --out={basename}"
            )
        else:
            os.system(
                f"eddy_{ed_type} --imain={input} --mask={mask} --acqp={acq} --index={index} --bvecs={bvecs} --bvals={bvals} \
                --topup={topup_data} --out={basename}" 
            )

        # Filtering and renaming the important output files
        Path(basename+'.nii.gz').rename(output+'.nii.gz')
        Path(basename+'.eddy_rotated_bvecs').rename(output+'.bvec')
        return output+'.nii.gz', output+'.bvec'

def rotate_Bmatrix(bvecs, bvals, output, matrix):
    """
    Rotates the b-matrix after the diffusion data has been 'moved' to the reference image.
    It assumes that all volumes are corrected for motion artifacts and are co-registered;
        therefore, the same transformation (rotation) matrix is applied to all b-vectors.
    """
    # Bvals are not rotated
    shutil.copyfile(bvals, output+'.bval')

    # Read and rotate non-zeros bvecs
    b_vals, b_vecs = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table(b_vals, b_vecs)
    vols = gtab.bvals.shape[0]
    v_trans, trans = np.loadtxt(matrix), list()
    for v in range(vols):
        if not gtab.b0s_mask[v]:
            trans.append(v_trans)
    gtab_corr = reorient_bvecs(gtab, trans)
    np.savetxt(output+'.bvec', gtab_corr.bvecs.T)
    return output+'.bvec', output+'.bval'

def functional_coregistration(output_directory, subject, session, strategy):
    """
    Register the BOLD images to MNI-anatomical space. The problem is that XCP-D does not recognise 
        the space to which anatomical and diffusion images are (AAL3 space). fmriprep can
        register to that space, but then XCP-D fails. 
    """
    # Anatomical directories
    anat_coregistered_dir = output_directory + 'co-registered/' + subject + '/'+ session +'/anat/'

    # Functional directories
    func_fmriprep_dir = output_directory + 'fmriprep/' + subject + '/'+ session +'/func/'
    func_xcpd_dir = output_directory + 'xcp_d/' + subject + '/'+ session +'/func/'
    coregistered_func_dir = output_directory + 'co-registered/' + subject + '/'+ session +'/func/'
    check_path(coregistered_func_dir)

    # Reference images
    ref_image = anat_coregistered_dir + subject + '_' + session + '_T1w.nii.gz'
    up_ref_img = upsample(ref_image, anat_coregistered_dir + subject + '_' + session + '_T1w_3mm', skip=False, factor=3)
    temporal_image = func_fmriprep_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_boldref.nii.gz'
    # Images to register
    input_images = [
        func_fmriprep_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
        func_xcpd_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-residual_bold.nii.gz',
        func_xcpd_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-residual_smooth_bold.nii.gz',
        func_fmriprep_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
    ]
    # Registered imges
    output_images = [
        coregistered_func_dir + subject + '_' + session + '_task-rest_bold_mask',
        coregistered_func_dir + subject + '_' + session + '_task-rest_bold_residual_' + strategy,
        coregistered_func_dir + subject + '_' + session + '_task-rest_bold_residual_smooth_' + strategy,
        coregistered_func_dir + subject + '_' + session + '_task-rest_bold_prepoc',
    ]
    # Register BOLD ref
    forget, tr_matrix = register_to_reference(
            input=temporal_image, output=coregistered_func_dir+subject+'-func2T1w', reference=up_ref_img, skip=False
        )
    os.system(f"rm {forget}")
    # Register imges
    for in_img, out_img in zip(input_images, output_images):
        register_to_reference(
                input=in_img, output=out_img, reference=up_ref_img, skip=False, matrix=tr_matrix, dim=4
            )
    os.system(f"rm {up_ref_img}")

    lesion_image = anat_coregistered_dir + subject + '_' + session + '_T1w_tumor.nii.gz'
    if not os.path.exists(lesion_image):
        lesion_image = anat_coregistered_dir + subject + '_' + session + '_T1w_stroke.nii.gz'
    if 'PAT' in subject:
        upsample(lesion_image, anat_coregistered_dir + subject + '_' + session + '_T1w_tumor_3mm', skip=False, factor=3)
        
    """ # Move fmriprep images
    img = func_fmriprep_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    new_img = coregistered_func_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    os.system(f"cp {img} {new_img}")
    img = func_fmriprep_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    new_img = coregistered_func_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    os.system(f"cp {img} {new_img}")
    
    # Move xcp-d images
    img = func_xcpd_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-residual_bold.nii.gz'
    new_img = coregistered_func_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-residual_bold.nii.gz'
    os.system(f"cp {img} {new_img}")
    img = func_xcpd_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-residual_smooth_bold.nii.gz'
    new_img = coregistered_func_dir + subject + '_' + session + '_task-rest_space-MNI152NLin2009cAsym_desc-residual_smooth_bold.nii.gz'
    os.system(f"cp {img} {new_img}") """