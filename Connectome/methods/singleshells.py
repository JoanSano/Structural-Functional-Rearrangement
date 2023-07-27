import os
import logging
import re

from utils.paths import get_files, output_directory
from utils.graph import GraphFromCSV
from utils.trac import tck2trk, upsample

def connectome_ss3t_csd(config, f, acronym):
    """
    Generate the connectome for a single file using SS3T-CSD. Using the atlas as a reference.
    """
    skip = config['skip']
    t_config = config['trac']
    skip_subjects = str(config['paths']['skip_subjects'])

    ### Files and directories related to the subject processed ###
    dataset_path = config['paths']['dataset_dir']
    output_dir, inter_dir = output_directory(f, dataset_path, acronym, custom_output=config['paths']['output_dir'])
    subject_ID, session, nii_dwi, mask_dwi, bval_dwi, bvec_dwi, json_dwi, nii_t1, mask_t1, lesion_t1 = get_files(f, config['lesion'])
    
    png_final_output = output_dir + subject_ID + '_' + session + '.png'
    if skip and os.path.exists(png_final_output):
        logging.info(" Skipping " + subject_ID + " in " + session + " PNG graph available in the Output Directory")
        return 

    if re.findall('...[0-9]+',subject_ID)[0] in skip_subjects:
        logging.info(" Skipping " + subject_ID + " in "  + session + " as specified in the config file")
    else: 
        ### Bias field homogeneities ###
        nii_dwi_bc = inter_dir+subject_ID+'_dwi_bc.mif'
        if skip and os.path.exists(nii_dwi_bc):
            logging.info(" " + subject_ID + " in " + session + " bc image already available... skipping")
        else:
            logging.info(" " + subject_ID + " in " + session + " bias field correction")
            os.system(f"dwibiascorrect ants {nii_dwi} {nii_dwi_bc} -fslgrad {bvec_dwi} {bval_dwi} -force -quiet")

        ### Run the 5-tissue-type segmentation ###
        if config['trac']['method'] == 'ACT' or config['trac']['seeding'] == 'gwmwi':
            act_5tt_seg = inter_dir + "5tt_seg.mif"
            lesion_t1_1mm = inter_dir + config["lesion"] + ".mif"
            if skip and os.path.exists(act_5tt_seg):
                logging.info(" " + subject_ID + " in " + session + " 5TT segmentation already available... skipping")
            else:
                logging.info(" " + subject_ID + " in " + session + " 5TT segmentation")
                os.system(f"5ttgen fsl {nii_t1} {act_5tt_seg} -mask {mask_t1} -nocrop -force -quiet")
                if not os.path.exists(act_5tt_seg):
                    logging.info(" " + subject_ID + " in " + session + " 5TT segmentation done with the -premasked option")
                    os.system(f"5ttgen fsl {nii_t1} {act_5tt_seg} -premasked -nocrop -force -quiet")
                if os.path.exists(lesion_t1):
                    upsample(lesion_t1, lesion_t1_1mm, factor=1)
                    os.system(f"5ttedit -path {lesion_t1_1mm} {act_5tt_seg} {act_5tt_seg} -force -quiet")

        ### Extract b0 + single shell ###
        nii_dwi_sshell, ss_bvec, ss_bval = inter_dir+'dwi_single_shell.mif', inter_dir+'bvec_single_shell.bvec', inter_dir+'bval_single_shell.bval'
        if skip and os.path.exists(nii_dwi_sshell):
            logging.info(" " + subject_ID + " in " + session + " Single Shell extraction already available... skipping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Single Shell extraction")
            os.system(f"dwiextract {nii_dwi_bc} {nii_dwi_sshell} -shells 0,{config['shell']} -fslgrad {bvec_dwi} {bval_dwi}\
                        -export_grad_fsl {ss_bvec} {ss_bval} -force -quiet")
            os.system(f"mrconvert {nii_dwi_sshell} -fslgrad {ss_bvec} {ss_bval} {nii_dwi_sshell} -force -quiet")

        ### Response Function Estimation ### 
        wm_res, gm_res, csf_res = inter_dir+'wm_response.txt', inter_dir+'gm_response.txt', inter_dir+'csf_response.txt'
        if skip and os.path.exists(wm_res) and os.path.exists(gm_res) and os.path.exists(csf_res):
            logging.info(" " + subject_ID + " in " + session + " Response functions already available... skiping")
        else:
            vox = inter_dir+"res_voxels.mif"
            logging.info(" " + subject_ID + " in " + session + " Estimating response functions")
            os.system(f"dwi2response dhollander {nii_dwi_sshell} {wm_res} {gm_res} {csf_res} -mask {mask_dwi}\
                        -fslgrad {ss_bvec} {ss_bval} -force -quiet -voxels {vox}")
        
        ### Run the reconstruction algorithm ###
        wm_fod, gm_fod, csf_fod = inter_dir+'wm_fod.mif', inter_dir+'gm_fod.mif', inter_dir+'csf_fod.mif'
        if skip and os.path.exists(wm_fod) and os.path.exists(gm_fod) and os.path.exists(csf_fod):
            logging.info(" " + subject_ID + " in " + session + " fODFs already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Reconstructing fODFs with SS3T-CSD")
            os.system(f"ss3t_csd_beta1_3T {nii_dwi_sshell} {wm_res} {wm_fod} {gm_res} {gm_fod} {csf_res} {csf_fod} \
                 -mask {mask_dwi} -force -quiet")

        ### Tissue component normalization ### (Aerts, et al. eNeuro 2018)
        wm_norm, gm_norm, csf_norm = inter_dir+'wm_fod_norm.mif', inter_dir+'gm_fod_norm.mif', inter_dir+'csf_fod_norm.mif'
        if skip and os.path.exists(wm_norm) and os.path.exists(gm_norm) and os.path.exists(csf_norm):
            logging.info(" " + subject_ID + " in " + session + " Normalized fODFs already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Normalizing fODFs")
            os.system(f"mtnormalise {wm_fod} {wm_norm} {gm_fod} {gm_norm} {csf_fod} {csf_norm} -mask {mask_dwi} -force -quiet")

        ### Generate the seeds for the tractography ###
        if config['trac']['seeding'] == "gmwmi": # GrawMatter-WhiteMatter interface
            seeding_mask = inter_dir+'seed_mask.mif'
            if skip and os.path.exists(seeding_mask):
                logging.info(" " + subject_ID + " in " + session + " Seeding mask already available... skiping")
            else:
                logging.info(" " + subject_ID + " in " + session + " Generating seeding masks")
                os.system(f"5tt2gmwmi {seeding_mask} -force -quiet")
            tck_seeding = "-seed_gmwmi " + seeding_mask
        elif config['trac']['seeding'] == "dynamic": # Dynamic seeding
            seeding_mask = wm_norm
            tck_seeding = "-seed_dynamic " + seeding_mask
        else:
            raise ValueError("Seedgin mechanism not implemented")

        ### Run tractography ###
        tck_file = output_dir + subject_ID + '_' + session + '_trac-' + t_config['streams'] + '.tck'
        tck_sift = output_dir + subject_ID + '_' + session + '_trac-' + t_config['streams'] + '_SIFT' + t_config['filtered'] + '.tck'
        if skip and os.path.exists(tck_sift):
            logging.info(" " + subject_ID + " in " + session + " SIFT Tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating tractogram")
            if config['trac']['method'] == 'ACT':
                os.system(f"tckgen -algorithm iFOD2 -act {act_5tt_seg} {tck_seeding} -backtrack -select {t_config['streams']} \
                        -seeds {t_config['seed_num']} -minlength {t_config['min_len']} -maxlength {t_config['max_len']} \
                        -fslgrad {bvec_dwi} {bval_dwi} -cutoff {2*float(config['trac']['cutoff'])} -force -quiet {wm_norm} {tck_file}")
            else:
                os.system(f"tckgen -algorithm iFOD2 {tck_seeding} -backtrack -select {t_config['streams']} \
                        -seeds {t_config['seed_num']} -minlength {t_config['min_len']} -maxlength {t_config['max_len']} \
                        -fslgrad {bvec_dwi} {bval_dwi} -mask {mask_dwi} -cutoff {config['trac']['cutoff']} \
                        -force -quiet {wm_norm} {tck_file}")

        ### Streamline filtering ###
        if skip and os.path.exists(tck_sift):
            logging.info(" " + subject_ID + " in " + session + " Filtered tractogram already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Filtering tractogram")
            if t_config['filtered'] == "":
                if config['trac']['method'] == 'ACT':
                    os.system(f"tcksift {tck_file} {wm_norm} {tck_sift} -act {act_5tt_seg} -force -quiet")
                else:
                    os.system(f"tcksift {tck_file} {wm_norm} {tck_sift} -force -quiet")
            else:
                filtered = int(int(t_config['streams'])*float(t_config['filtered'])/100)
                if config['trac']['method'] == 'ACT':
                    os.system(f"tcksift {tck_file} {wm_norm} {tck_sift} -act {act_5tt_seg} -term_number {filtered} -force -quiet")
                else:
                    os.system(f"tcksift {tck_file} {wm_norm} {tck_sift} -term_number {filtered} -force -quiet")
                    
        if t_config['save_trk']:
            logging.info(" " + subject_ID + " in " + session + " Converting to .trk")
            tck2trk(nii_t1, tck_sift)

        ### Generate Connectome ###
        cm_file = output_dir + subject_ID + '_' + session + '_CM.csv'
        cm2tck = output_dir + subject_ID + '_' + session + '_cm2trac.txt'
        if config["space"] == 'MNI':
            atlas_path = config["paths"]['atlas_path']
        else:
            atlas_path = config["paths"]['atlas_path'] + subject_ID + '/' + session + '/atlas/' + subject_ID + '_' + session + '_T1w_labels.nii.gz'
        if skip and os.path.exists(cm_file):
            logging.info(" " + subject_ID + " in " + session + " Connectome already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Generating connectome")
            os.system(f"tck2connectome {tck_sift} {atlas_path} {cm_file} \
                        -symmetric -zero_diagonal -out_assignments {cm2tck} -force -quiet")

        ### Structural connectivity stats ###
        if skip and os.path.exists(png_final_output):
            logging.info(" " + subject_ID + " in " + session + " Connectome stats already available... skiping")
        else:
            logging.info(" " + subject_ID + " in " + session + " Connectome stats")
            cm_file = output_dir + subject_ID + '_' + session + '_CM.csv'
            sg = GraphFromCSV(cm_file, subject_ID+'_'+session, output_dir)
            sg.flatten_graph(save=True) 
            sg.process_graph()

        ### Remove .tck heavy file ###
        if not t_config["keep_full_tck"] and os.path.exists(tck_file):
            os.remove(tck_file)

        ### Remove intermediates ###
        if config["delete"]["response_funcs"]:
            os.system(f"rm {wm_res} {gm_res} {csf_res} {vox}")
        if config["delete"]["fODFs"]:
            os.system(f"rm {wm_fod} {gm_fod} {csf_fod} {gm_norm} {csf_norm}")
        if config["delete"]["WM_fODF"]:
            os.system(f"rm {wm_norm}")
        if config["delete"]["seeding_mask"] and os.path.exists(seeding_mask):
            os.system(f"rm {seeding_mask}")
        if config["delete"]["bias_corr"]:
            os.system(f"rm {nii_dwi_bc}")
        if config["delete"]["seg_5tt"]:
            os.system(f"rm {act_5tt_seg}")
        if config["delete"]["single_shells"]:
            os.system(f"rm {nii_dwi_sshell} {ss_bvec} {ss_bval}")
        if config["delete"]["lesion_1mm"] and os.path.exists(lesion_t1_1mm):
            os.system(f"rm {lesion_t1_1mm}")

if __name__ == '__main__':
    pass