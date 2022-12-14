import os
import logging
import re

from utils.paths import get_files, output_directory
from utils.graph import GraphFromCSV
from utils.trac import tck2trk, lesion_deletion, upsample, merge_fods

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
        if os.path.exists(tumor_t1) and not os.path.exists(act_5tt_seg_pathological):
            os.system(f"5ttedit -path {tumor_t1} {act_5tt_seg} {act_5tt_seg_pathological} -force -quiet ")

        ####################################
        ### Reconstruction INSIDE oedema ###
        ####################################
        ### Extract b0 + single shell ###
        nii_dwi_sshell, ss_bvec, ss_bval = inter_dir+'dwi_single_shell.mif', inter_dir+'bvec_single_shell.bvec', inter_dir+'bval_single_shell.bval'
        if skip and os.path.exists(nii_dwi_sshell):
            logging.info(" " + subject_ID + " in " + session + " Single Shell extraction already available... skipping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Single Shell extraction")
            os.system(f"dwiextract {nii_dwi_bc} {nii_dwi_sshell} -shells 0,{config['shell']} -fslgrad {bvec_dwi} {bval_dwi}\
                        -export_grad_fsl {ss_bvec} {ss_bval} -force -quiet ")
            os.system(f"mrconvert {nii_dwi_sshell} -fslgrad {ss_bvec} {ss_bval} {nii_dwi_sshell} -force -quiet ")

        ### Response Function Estimation ### 
        oedema_wm_res, oedema_gm_res, oedema_csf_res = inter_dir+'oedema_wm_response.txt', inter_dir+'oedema_gm_response.txt', inter_dir+'oedema_csf_response.txt'
        oedema_vox = inter_dir+"oedema_res_voxels.mif"
        if skip and os.path.exists(oedema_wm_res) and os.path.exists(oedema_gm_res) and os.path.exists(oedema_csf_res):
            logging.info(" " + subject_ID + " in " + session + " Response functions inside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Estimating response functions inside lesion")
            os.system(f"dwi2response dhollander {nii_dwi_sshell} {oedema_wm_res} {oedema_gm_res} {oedema_csf_res} -mask {mask_dwi}\
                        -fslgrad {ss_bvec} {ss_bval} -force -quiet -voxels {oedema_vox} ")
        
        ### Run the reconstruction algorithm ###
        oedema_wm_fod, oedema_gm_fod, oedema_csf_fod = inter_dir+'oedema_wm_fod.mif', inter_dir+'oedema_gm_fod.mif', inter_dir+'oedema_csf_fod.mif'
        oedema_wm_norm, oedema_gm_norm, oedema_csf_norm = inter_dir+'oedema_wm_fod_norm.mif', inter_dir+'oedema_gm_fod_norm.mif', inter_dir+'oedema_csf_fod_norm.mif'
        if skip and os.path.exists(oedema_wm_norm):
            logging.info(" " + subject_ID + " in " + session + " fODFs inside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Reconstructing fODFs with SS3T-CSD inside lesion")
            os.system(f"ss3t_csd_beta1_3T {nii_dwi_sshell} {oedema_wm_res} {oedema_wm_fod} {oedema_gm_res} {oedema_gm_fod} {oedema_csf_res} {oedema_csf_fod} \
                 -mask {tumor_dwi} -force -quiet ")
        
        ### Tissue component normalization ### (Aerts, et al. eNeuro 2018)
        if skip and os.path.exists(oedema_wm_norm):
            logging.info(" " + subject_ID + " in " + session + " Normalized fODFs inside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Normalizing fODFs inside lesion")
            os.system(f"mtnormalise {oedema_wm_fod} {oedema_wm_norm} {oedema_gm_fod} {oedema_gm_norm} {oedema_csf_fod} {oedema_csf_norm} -mask {tumor_dwi} -force -quiet ")

        #####################################
        ### Reconstruction OUTSIDE oedema ###
        #####################################
        ### Response Function Estimation ### 
        wm_res, gm_res, csf_res = inter_dir+'wm_response.txt', inter_dir+'gm_response.txt', inter_dir+'csf_response.txt'
        vox = inter_dir+"res_voxels.mif"
        if skip and os.path.exists(wm_res) and os.path.exists(gm_res) and os.path.exists(csf_res):
            logging.info(" " + subject_ID + " in " + session + " Response functions outside lesion already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Estimating response functions outside lesion")
            os.system(f"dwi2response dhollander {nii_dwi_bc} {wm_res} {gm_res} {csf_res} -mask {mask_dwi}\
                        -fslgrad {bvec_dwi} {bval_dwi} -force -quiet -voxels {vox} ")

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
            oedema_norm_nii, norm_nii = inter_dir+'oedema_wm_fod_norm.nii.gz', inter_dir+'wm_fod_norm.nii.gz'
            os.system(f"mrconvert {oedema_wm_norm} {oedema_norm_nii} -quiet -force ")
            os.system(f"mrconvert {wm_norm} {norm_nii} -quiet -force ")
            wm_merged = merge_fods(norm_nii, oedema_norm_nii, inter_dir)

        ############################
        ### Fiber Tracking steps ###
        ############################
        ### Generate the seeding masks for the tractography ###
        if config['trac']['seeding'] == "random":
            oedema_tck_seeding = "-seed_image " + tumor_dwi
            healthy_tck_seeding = "-seed_image " + mask_dwi
        elif config['trac']['seeding'] == "dynamic": # Dynamic seeding
            oedema_seeding_mask = oedema_wm_norm 
            oedema_tck_seeding = "-seed_dynamic " + oedema_seeding_mask
            healthy_seeding_mask = wm_norm
            healthy_tck_seeding = "-seed_dynamic " + healthy_seeding_mask
        else:
            raise ValueError("Seeding mechanism not implemented")

        ### Run tractography through lesion without ACT ###
        oedema_tck_file = output_dir + subject_ID + '_' + session + '_trac' + t_config['streams'] + '_lesion.tck'
        oedema_tck_sift = output_dir + subject_ID + '_' + session + '_trac' + t_config['streams'] + '_lesion_SIFT.tck'
        if skip and os.path.exists(oedema_tck_sift):
            logging.info(" " + subject_ID + " in " + session + " SIFT lesion tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating and filtering tractogram through lesion")
            # We use the merged fod files to capture tracts that pass through the lesion
            # We seed only inside the lesion
            # We filter with fODFs from all over the brain without a fixed number of final streams
            os.system(f"tckgen -algorithm iFOD2 {oedema_tck_seeding} -backtrack -select {t_config['streams']} \
                -seeds {t_config['seed_num']} -minlength {t_config['min_len']} -maxlength {t_config['max_len']} \
                -fslgrad {bvec_dwi} {bval_dwi} -mask {whole_t1_mask} -cutoff {config['trac']['cutoff']} -force -quiet  {wm_merged} {oedema_tck_file}")    
            os.system(f"tcksift {oedema_tck_file} {wm_merged} {oedema_tck_sift} -act {act_5tt_seg_pathological} -force -quiet ") 

        ### Run tractography outside lesion with ACT ###
        healthy_tck_file = output_dir + subject_ID + '_' + session + '_trac' + t_config['streams'] + '_healthy.tck'
        healthy_tck_sift = output_dir + subject_ID + '_' + session + '_trac' + t_config['streams'] + '_healthy_SIFT' + t_config['filtered'] + '.tck'
        if skip and os.path.exists(healthy_tck_sift):
            logging.info(" " + subject_ID + " in " + session + " SIFT healthy tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating and filtering tractogram outside lesion")
            # We use thefod file made outside the lesion
            # We seed only outside the lesion
            # We filter only with fODFs outside the lesion
            os.system(f"tckgen -algorithm iFOD2 -act {act_5tt_seg} {healthy_tck_seeding} -backtrack -select {t_config['streams']} \
                    -seeds {t_config['seed_num']} -minlength {t_config['min_len']} -maxlength {t_config['max_len']} \
                    -fslgrad {bvec_dwi} {bval_dwi} -force -quiet  {wm_norm} {healthy_tck_file}")
            if t_config['filtered'] == "":
                os.system(f"tcksift {healthy_tck_file} {wm_norm} {healthy_tck_sift} -act {act_5tt_seg} -force -quiet ")
            else:
                filtered = int(int(t_config['streams'])*float(t_config['filtered'])/100)
                os.system(f"tcksift {healthy_tck_file} {wm_norm} {healthy_tck_sift} -act {act_5tt_seg} -term_number {filtered} -force -quiet ")

        ### Merge tractograms and filter them to avoid innecessary repetitions ###
        merged_tck_file = output_dir + subject_ID + '_' + session + '_trac' + t_config['streams'] + '.tck'
        merged_tck_sift = output_dir + subject_ID + '_' + session + '_trac' + t_config['streams'] + '_SIFT' + t_config['filtered'] + '.tck'
        if skip and os.path.exists(merged_tck_sift):
            logging.info(" " + subject_ID + " in " + session + " SIFT merged tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Merging and filtering tractogram")
            # We filter the merged UNFILTERED tractograms with the whole brain fODFs
            os.system(f"tckedit {healthy_tck_file} {oedema_tck_file} {merged_tck_file} -force -quiet")    
            if t_config['filtered'] == "":
                os.system(f"tcksift {merged_tck_file} {wm_merged} {merged_tck_sift} -act {act_5tt_seg_pathological} -force -quiet ")
            else:
                filtered = int(int(t_config['streams'])*float(t_config['filtered'])/100)
                os.system(f"tcksift {merged_tck_file} {wm_merged} {merged_tck_sift} -act {act_5tt_seg_pathological} -term_number {filtered} -force -quiet ")  

        if t_config['save_trk']:
            logging.info(" " + subject_ID + " in " + session + " Converting to .trk")
            tck2trk(nii_t1, healthy_tck_sift)
            tck2trk(nii_t1, oedema_tck_sift)
            tck2trk(nii_t1, merged_tck_sift)

        ### Generate Connectomes ###
        healthy_cm_file = output_dir + subject_ID + '_' + session + '_healthy_CM.csv'
        healthy_cm2tck = output_dir + subject_ID + '_' + session + '_healthy_cm2trac.txt'
        oedema_cm_file = output_dir + subject_ID + '_' + session + '_lesion_CM.csv'
        oedema_cm2tck = output_dir + subject_ID + '_' + session + '_lesion_cm2trac.txt'
        merged_cm_file = output_dir + subject_ID + '_' + session + '_CM.csv'
        merged_cm2tck = output_dir + subject_ID + '_' + session + '_cm2trac.txt'
        if config["space"] == 'MNI':
            atlas_path = config["paths"]['atlas_path']
        else:
            atlas_path = config["paths"]['atlas_path'] + subject_ID + '/' + session + '/atlas/' + subject_ID + '_' + session + '_T1w_labels.nii.gz'
        if skip and os.path.exists(healthy_cm_file):
            logging.info(" " + subject_ID + " in " + session + " Healthy connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating healthy connectome")
            os.system(f"tck2connectome {healthy_tck_sift} {atlas_path} {healthy_cm_file} \
                        -symmetric -zero_diagonal -out_assignments {healthy_cm2tck} -force -quiet ")
        if skip and os.path.exists(oedema_cm_file):
            logging.info(" " + subject_ID + " in " + session + " Lesion connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating lesion connectome")
            os.system(f"tck2connectome {oedema_tck_sift} {atlas_path} {oedema_cm_file} \
                        -symmetric -zero_diagonal -out_assignments {oedema_cm2tck} -force -quiet ")
        if skip and os.path.exists(merged_cm_file):
            logging.info(" " + subject_ID + " in " + session + " Merged connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating merged connectome")
            os.system(f"tck2connectome {merged_tck_sift} {atlas_path} {merged_cm_file} \
                        -symmetric -zero_diagonal -out_assignments {merged_cm2tck} -force -quiet ")

        ### Structural connectivity stats ###
        if skip and os.path.exists(output_dir + '_' + subject_ID + session + '.png'):
            logging.info(" " + subject_ID + " in " + session + " Connectome stats already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Connectome stats")
            sg = GraphFromCSV(healthy_cm_file, subject_ID+'_'+session+'_healthy', output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()
            sg = GraphFromCSV(oedema_cm_file, subject_ID+'_'+session+'_lesion', output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()
            sg = GraphFromCSV(merged_cm_file, subject_ID+'_'+session, output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()

        ### Remove .tck heavy file ###
        if not t_config["keep_full_tck"] and os.path.exists(healthy_tck_file):
            os.remove(healthy_tck_file)
        if not t_config["keep_full_tck"] and os.path.exists(oedema_tck_file):
            os.remove(oedema_tck_file)
        if not t_config["keep_full_tck"] and os.path.exists(merged_tck_file):
            os.remove(merged_tck_file)

        ### Remove intermediates ###
        if config["delete"]["response_funcs"]:
            os.system(f"rm {wm_res} {gm_res} {csf_res} {vox} {oedema_wm_res} {oedema_gm_res} {oedema_csf_res} {oedema_vox}")
        if config["delete"]["fODFs"]:
            os.system(f"rm {wm_fod} {gm_fod} {csf_fod} {gm_norm} {csf_norm} \
                {oedema_wm_fod} {oedema_gm_fod} {oedema_csf_fod} {oedema_gm_norm} {oedema_csf_norm}")
        if config["delete"]["WM_fODF"]:
            os.system(f"rm {wm_norm} {oedema_wm_norm} {wm_merged}")
        if config["delete"]["seeding_mask"] and os.path.exists(healthy_seeding_mask):
            os.system(f"rm {healthy_seeding_mask}")
        if config["delete"]["seeding_mask"] and os.path.exists(oedema_seeding_mask):
            os.system(f"rm {oedema_seeding_mask}")
        if config["delete"]["bias_corr"]:
            os.system(f"rm {nii_dwi_bc}")
        if config["delete"]["seg_5tt"]:
            os.system(f"rm {act_5tt_seg} {act_5tt_seg_pathological}")
        if config["delete"]["single_shells"]:
            os.system(f"rm {nii_dwi_sshell} {ss_bvec} {ss_bval}")

if __name__ == '__main__':
    pass