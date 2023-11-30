import os
import logging
import re

from utils.paths import get_files, output_directory
from utils.graph import GraphFromCSV
from utils.trac import *

def merge_connectivity(healthy_cm_file, lesion_cm_file, merged_cm_filename, strategy='greedy', sift2_weights_healthy=None, sift2_weights_lesion=None):
    """
    Implements a strategy to merge the connectivity matrices.
    """
    import numpy as np
    import itertools

    # Load matrices
    healthy_cm = np.genfromtxt(healthy_cm_file, delimiter=',')
    lesion_cm = np.genfromtxt(lesion_cm_file, delimiter=',')
    
    # Check whether matrices are mergeable
    assert np.shape(healthy_cm) == np.shape(lesion_cm)
    nodes = np.shape(healthy_cm)[0]
    
    # Implement merging strategy
    merged_cm = np.zeros((nodes,nodes))
    for (i,j) in itertools.product(range(nodes), repeat=2): # Iterate over edges
        if strategy=='greedy':
            merged_cm[i,j] = np.max([
                healthy_cm[i,j], lesion_cm[i,j]
            ])
        else:
            raise Exception('Merging strategy not implemented')
        
    # Save and return the file name
    np.savetxt(merged_cm_filename, merged_cm, delimiter=',')

def hybrid(config, f, acronym):
    """
    Generate the connectome for a single file using SS3T-CSD inside pathological tissue 
        and MSMT-CSD outside the lesion mask. 
    Using the atlas as a reference.
    """
    skip = config['skip']
    t_config = config['trac']
    skip_subjects = str(config['paths']['skip_subjects'])

    ### Files and directories related to the subject processed ###
    dataset_path = config['paths']['dataset_dir']
    output_dir, inter_dir = output_directory(f, dataset_path, acronym, custom_output=config['paths']['output_dir'])
    subject_ID, session, nii_dwi, mask_dwi, bval_dwi, bvec_dwi, json_dwi, nii_t1, mask_t1, tumor_t1 = get_files(f, config['lesion'])
    
    ### Preparing the binary masks ###
    whole_t1_mask = mask_t1 
    mask_t1 = lesion_deletion(mask_t1, tumor_t1, inter_dir)
    tumor_dwi = upsample(tumor_t1, inter_dir+'TUMOR_'+mask_dwi.split('/')[-1], factor=1.5)
    mask_dwi = lesion_deletion(mask_dwi, tumor_dwi, inter_dir)

    if re.findall('...[0-9]+',subject_ID)[0] in skip_subjects:
        logging.info(" Skipping " + subject_ID + session)
    else: 
        
        ### Bias field homogeneities ###
        nii_dwi_bc = inter_dir+subject_ID+'_dwi_bc.mif'
        if skip and os.path.exists(nii_dwi_bc):
            logging.info(" " + subject_ID + " in " + session + " bc image already available... skipping")
        else:
            logging.info(" " + subject_ID + " in " + session + " bias field correction")
            os.system(f"dwibiascorrect ants {nii_dwi} {nii_dwi_bc} -fslgrad {bvec_dwi} {bval_dwi} -force -quiet ")
        
        ### Run the 5-tissue-type segmentation ###
        act_5tt_seg = inter_dir + "5tt_seg.mif"
        act_5tt_seg_pathological = inter_dir + "5tt_seg_pathological.mif"
        if skip and os.path.exists(act_5tt_seg):
            logging.info(" " + subject_ID + " in " + session + " 5TT segmentation already available... skipping")
        else:
            logging.info(" " + subject_ID + " in " + session + " 5TT segmentation")
            os.system(f"5ttgen fsl {nii_t1} {act_5tt_seg} -mask {mask_t1} -nocrop -force -quiet ")
            if not os.path.exists(act_5tt_seg):
                logging.info(" " + subject_ID + " in " + session + " 5TT segmentation done with the -premasked option")
                os.system(f"5ttgen fsl {nii_t1} {act_5tt_seg} -premasked -nocrop -force -quiet")
        if skip and os.path.exists(act_5tt_seg_pathological):
            logging.info(" " + subject_ID + " in " + session + " 5TT with lesion already available... skipping")
        else:
            os.system(f"5ttedit -path {tumor_t1} {act_5tt_seg} {act_5tt_seg_pathological} -force -quiet ")
        
        ### Response Function Estimation - using only healthy tissue ### 
        wm_res, gm_res, csf_res = inter_dir+'wm_response.txt', inter_dir+'gm_response.txt', inter_dir+'csf_response.txt'
        vox = inter_dir+"res_voxels.mif"
        if skip and os.path.exists(wm_res) and os.path.exists(gm_res) and os.path.exists(csf_res):
            logging.info(" " + subject_ID + " in " + session + " Response functions outside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Estimating response functions outside lesion")
            os.system(f"dwi2response dhollander {nii_dwi_bc} {wm_res} {gm_res} {csf_res} -mask {mask_dwi}\
                        -fslgrad {bvec_dwi} {bval_dwi} -force -quiet -voxels {vox} ")
            
        lesion_wm_res, lesion_gm_res, lesion_csf_res = inter_dir+'lesion_wm_response.txt', inter_dir+'lesion_gm_response.txt', inter_dir+'lesion_csf_response.txt'
        if skip and os.path.exists(lesion_wm_res) and os.path.exists(lesion_gm_res) and os.path.exists(lesion_csf_res):
            logging.info(" " + subject_ID + " in " + session + " Response functions inside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Extracting the single shell response function")
            extract_response_shell(wm_res, lesion_wm_res, gm_res, lesion_gm_res, csf_res, lesion_csf_res, config_shell=config['shell'])

        ####################################
        ### Reconstruction INSIDE lesion ###
        ####################################
        ### Extract b0 + single shell ###
        nii_dwi_sshell, ss_bvec, ss_bval = inter_dir+'dwi_single_shell.mif', inter_dir+'bvec_single_shell.bvec', inter_dir+'bval_single_shell.bval'
        if skip and os.path.exists(nii_dwi_sshell):
            logging.info(" " + subject_ID + " in " + session + " Single Shell extraction already available... skipping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Single Shell extraction")
            os.system(f"dwiextract {nii_dwi_bc} {nii_dwi_sshell} -shells 0,{config['shell']['b_val']} -fslgrad {bvec_dwi} {bval_dwi}\
                        -export_grad_fsl {ss_bvec} {ss_bval} -force -quiet ")
            os.system(f"mrconvert {nii_dwi_sshell} -fslgrad {ss_bvec} {ss_bval} {nii_dwi_sshell} -force -quiet ")
        
        ### Run the reconstruction algorithm ###
        lesion_wm_fod, lesion_gm_fod, lesion_csf_fod = inter_dir+'lesion_wm_fod.mif', inter_dir+'lesion_gm_fod.mif', inter_dir+'lesion_csf_fod.mif'
        lesion_wm_norm, lesion_gm_norm, lesion_csf_norm = inter_dir+'lesion_wm_fod_norm.mif', inter_dir+'lesion_gm_fod_norm.mif', inter_dir+'lesion_csf_fod_norm.mif'
        if skip and os.path.exists(lesion_wm_norm):
            logging.info(" " + subject_ID + " in " + session + " fODFs inside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Reconstructing fODFs with SS3T-CSD inside lesion")
            os.system(f"ss3t_csd_beta1_3T {nii_dwi_sshell} {lesion_wm_res} {lesion_wm_fod} {lesion_gm_res} {lesion_gm_fod} {lesion_csf_res} {lesion_csf_fod} \
                 -mask {tumor_dwi} -force -quiet ")
        
        ### Tissue component normalization ### (Aerts, et al. eNeuro 2018)
        if skip and os.path.exists(lesion_wm_norm):
            logging.info(" " + subject_ID + " in " + session + " Normalized fODFs inside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Normalizing fODFs inside lesion")
            os.system(f"mtnormalise {lesion_wm_fod} {lesion_wm_norm} {lesion_gm_fod} {lesion_gm_norm} {lesion_csf_fod} {lesion_csf_norm} -mask {tumor_dwi} -force -quiet ")
        
        #####################################
        ### Reconstruction OUTSIDE lesion ###
        #####################################
        ### Run the reconstruction algorithm ###
        wm_fod, gm_fod, csf_fod = inter_dir+'wm_fod.mif', inter_dir+'gm_fod.mif', inter_dir+'csf_fod.mif'
        wm_norm, gm_norm, csf_norm = inter_dir+'wm_fod_norm.mif', inter_dir+'gm_fod_norm.mif', inter_dir+'csf_fod_norm.mif'
        if skip and os.path.exists(wm_norm): 
            logging.info(" " + subject_ID + " in " + session + "fODFs outside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Reconstructing fODFs with MSMT-CSD outside lesion")
            os.system(f"dwi2fod msmt_csd {nii_dwi_bc} {wm_res} {wm_fod} {gm_res} {gm_fod} {csf_res} {csf_fod} \
                        -mask {mask_dwi} -fslgrad {bvec_dwi} {bval_dwi} -force -quiet ")

        ### Tissue component normalization ### (Aerts, et al. eNeuro 2018)
        if skip and os.path.exists(wm_norm): 
            logging.info(" " + subject_ID + " in " + session + " Normalized fODFs outside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Normalizing fODFs outside lesion")
            os.system(f"mtnormalise {wm_fod} {wm_norm} {gm_fod} {gm_norm} {csf_fod} {csf_norm} -mask {mask_dwi} -force -quiet ")

        ############################
        ### Merging WM FOD files ###
        ############################
        if skip and os.path.exists(inter_dir+'FODs.mif'):
             logging.info(" " + subject_ID + " in " + session + " Merged fODFs reconstructions already available... skiping")
             wm_merged = inter_dir + 'FODs.mif'
        else:
            logging.info(" " + subject_ID + " in " + session + " Merging WM fODFs reconstructions")
            lesion_norm_nii, norm_nii = inter_dir+'lesion_wm_fod_norm.nii.gz', inter_dir+'wm_fod_norm.nii.gz'
            os.system(f"mrconvert {lesion_wm_norm} {lesion_norm_nii} -quiet -force ")
            os.system(f"mrconvert {wm_norm} {norm_nii} -quiet -force ")
            wm_merged = merge_fods(norm_nii, lesion_norm_nii, inter_dir)
        
        ###############
        ### Seeding ###
        ###############
        ### Generate the seeding masks for the tractography ###
        if config['trac']['seeding'] == "random":
            lesion_tck_seeding = "-seed_image " + tumor_dwi
            healthy_tck_seeding = "-seed_image " + mask_dwi
        elif config['trac']['seeding'] == "dynamic": # Dynamic seeding
            lesion_seeding_mask = lesion_wm_norm 
            lesion_tck_seeding = "-seed_dynamic " + lesion_seeding_mask
            healthy_seeding_mask = wm_norm
            healthy_tck_seeding = "-seed_dynamic " + healthy_seeding_mask
        else:
            raise ValueError("Seeding mechanism not implemented")

        #######################################
        ### Obtaining number of streamlines ###
        #######################################
        num_streams_lesion = inter_dir+subject_ID+'_lesion-streamlines.txt'
        if skip and os.path.exists(num_streams_lesion):
            logging.info(" " + subject_ID + " in " + session + " Streamline counts already available... skiping")
            streams_lesion = read_healthy_streamlines_from_lesion(num_streams_lesion)
        else:
            logging.info(" " + subject_ID + " in " + session + " Counting number of streamlines in the healthy cohort")
            streams_lesion = extract_healthy_streamlines_from_lesion(
                config["shell"]["healthy_dir"], tumor_t1, num_streams_lesion, subject_ID
            )
        streams_outside = str(int(float(t_config['streams'])-streams_lesion))
        streams_lesion = str(streams_lesion)
        
        ##############################
        ### Tracking within lesion ###
        ##############################
        ### Run tractography through lesion without ACT ###
        lesion_tck_file = output_dir + subject_ID + '_' + session + '_trac-' + streams_lesion + '_lesion.tck'
        if skip and os.path.exists(lesion_tck_file):
            logging.info(" " + subject_ID + " in " + session + " Lesion tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating and filtering tractogram through lesion")
            # We use the merged fod files to capture tracts that pass through the lesion
            # We seed only inside the lesion
            os.system(f"tckgen -algorithm iFOD2 {lesion_tck_seeding} -backtrack -select {streams_lesion} \
                -minlength {t_config['min_len']} -maxlength {t_config['max_len']} \
                -fslgrad {bvec_dwi} {bval_dwi} -mask {whole_t1_mask} -cutoff {config['trac']['cutoff']} \
                -force -quiet  {wm_merged} {lesion_tck_file}")    
        
        ### SIF2 to match the lesion underlying diffusion signal ###
        lesion_weights_sift = output_dir + subject_ID + '_' + session + '_trac-' + streams_lesion + '_SIFT2-weights_tkh-' + t_config['sift2_tikhonov'] + '_tv-' + t_config['sift2_tv'] + '_lesion.txt'
        if skip and os.path.exists(lesion_weights_sift):
            logging.info(" " + subject_ID + " in " + session + " Filtered tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Filtering tractogram")
            # We filter with fODFs from the lesion area 
            os.system(f"tcksift2 {lesion_tck_file} {lesion_wm_norm} {lesion_weights_sift} -fd_scale_gm \
                    -reg_tikhonov {t_config['sift2_tikhonov']} -reg_tv {t_config['sift2_tv']} -force -quiet")
        
        ###############################
        ### Tracking without lesion ###
        ###############################
        ### Run tractography outside lesion with ACT ###
        healthy_tck_file = output_dir + subject_ID + '_' + session + '_trac-' + streams_outside + '_healthy.tck'
        if skip and os.path.exists(healthy_tck_file):
            logging.info(" " + subject_ID + " in " + session + " Healthy tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating tractogram outside lesion")
            # We use the FOD file made outside the lesion
            # We seed only outside the lesion
            # We filter only with fODFs outside the lesion
            os.system(f"tckgen -algorithm iFOD2 -act {act_5tt_seg} {healthy_tck_seeding} -backtrack -select {streams_outside} \
                    -seeds {t_config['seed_num']} -minlength {t_config['min_len']} -maxlength {t_config['max_len']} \
                    -fslgrad {bvec_dwi} {bval_dwi} -cutoff {2*float(config['trac']['cutoff'])} -force -quiet  {wm_norm} {healthy_tck_file}")
            
        ### SIFT2 to match the healthy underlying diffusion signal ###
        healthy_weights_sift = output_dir + subject_ID + '_' + session + '_trac-' + streams_outside + '_SIFT2-weights_tkh-' + t_config['sift2_tikhonov'] + '_tv-' + t_config['sift2_tv'] + '_healthy.txt'
        if skip and os.path.exists(healthy_weights_sift):
            logging.info(" " + subject_ID + " in " + session + " Filtered tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Filtering tractogram")
            os.system(f"tcksift2 {healthy_tck_file} {wm_norm} {healthy_weights_sift} -act {act_5tt_seg} -fd_scale_gm \
                    -reg_tikhonov {t_config['sift2_tikhonov']} -reg_tv {t_config['sift2_tv']} -force -quiet")

        # To .trk format
        if t_config['save_trk']:
            logging.info(" " + subject_ID + " in " + session + " Converting to .trk")
            tck2trk(nii_t1, healthy_tck_file)
            tck2trk(nii_t1, lesion_tck_file)

        ###########################################
        ### Merging strategies and connectomics ###
        ###########################################
        ### Generate Connectomes ###
        healthy_cm_file = output_dir + subject_ID + '_' + session + '_healthy_CM.csv'
        healthy_cm2tck = output_dir + subject_ID + '_' + session + '_healthy_cm2trac.txt'
        lesion_cm_file = output_dir + subject_ID + '_' + session + '_lesion_CM.csv'
        lesion_cm2tck = output_dir + subject_ID + '_' + session + '_lesion_cm2trac.txt'
        merged_cm_file = output_dir + subject_ID + '_' + session + '_CM.csv'
        #merged_cm2tck = output_dir + subject_ID + '_' + session + '_cm2trac.txt'
        if config["space"] == 'MNI':
            atlas_path = config["paths"]['atlas_path']
        else:
            atlas_path = config["paths"]['atlas_path'] + subject_ID + '/' + session + '/atlas/' + subject_ID + '_' + session + '_T1w_labels.nii.gz'
        
        ### Healthy Connectome ###
        if skip and os.path.exists(healthy_cm_file):
            logging.info(" " + subject_ID + " in " + session + " Healthy connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating healthy connectome")
            os.system(f"tck2connectome {healthy_tck_file} {atlas_path} {healthy_cm_file} -tck_weights_in {healthy_weights_sift} \
                        -symmetric -zero_diagonal -out_assignments {healthy_cm2tck} -force -quiet ")
        ### Lesion Connectome ###
        if skip and os.path.exists(lesion_cm_file):
            logging.info(" " + subject_ID + " in " + session + " Lesion connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating lesion connectome")
            os.system(f"tck2connectome {lesion_tck_file} {atlas_path} {lesion_cm_file} -tck_weights_in {lesion_weights_sift} \
                        -symmetric -zero_diagonal -out_assignments {lesion_cm2tck} -force -quiet ")
        ### Merging Connectomes ###
        if skip and os.path.exists(merged_cm_file):
            logging.info(" " + subject_ID + " in " + session + " Merged connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating merged connectome")
            # TODO: Matrix merging strategy
                # 1. Greedy approach
                # 2. Another alternative? Difficult to justify
            merge_connectivity(healthy_cm_file, lesion_cm_file, merged_cm_file, strategy='greedy')

        ### Structural connectivity stats ###
        if skip and os.path.exists(output_dir + '_' + subject_ID + session + '.png'):
            logging.info(" " + subject_ID + " in " + session + " Connectome stats already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Connectome stats")
            sg = GraphFromCSV(healthy_cm_file, subject_ID+'_'+session+'_healthy', output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()
            sg = GraphFromCSV(lesion_cm_file, subject_ID+'_'+session+'_lesion', output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()
            sg = GraphFromCSV(merged_cm_file, subject_ID+'_'+session, output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()

        ### Remove .tck heavy file ###
        if not t_config["keep_full_tck"] and os.path.exists(healthy_tck_file):
            os.remove(healthy_tck_file)
        if not t_config["keep_full_tck"] and os.path.exists(lesion_tck_file):
            os.remove(lesion_tck_file)

        ### Remove intermediates ###
        if config["delete"]["response_funcs"]:
            os.system(f"rm {wm_res} {gm_res} {csf_res} {vox} {lesion_wm_res} {lesion_gm_res} {lesion_csf_res}")
        if config["delete"]["fODFs"]:
            os.system(f"rm {wm_fod} {gm_fod} {csf_fod} {gm_norm} {csf_norm} \
                {lesion_wm_fod} {lesion_gm_fod} {lesion_csf_fod} {lesion_gm_norm} {lesion_csf_norm}")
        if config["delete"]["WM_fODF"]:
            os.system(f"rm {wm_norm} {lesion_wm_norm} {wm_merged}")
        if config["delete"]["seeding_mask"] and os.path.exists(healthy_seeding_mask):
            os.system(f"rm {healthy_seeding_mask}")
        if config["delete"]["seeding_mask"] and os.path.exists(lesion_seeding_mask):
            os.system(f"rm {lesion_seeding_mask}")
        if config["delete"]["bias_corr"]:
            os.system(f"rm {nii_dwi_bc}")
        if config["delete"]["seg_5tt"]:
            os.system(f"rm {act_5tt_seg} {act_5tt_seg_pathological}")
        if config["delete"]["single_shells"]:
            os.system(f"rm {nii_dwi_sshell} {ss_bvec} {ss_bval}")

if __name__ == '__main__':
    pass