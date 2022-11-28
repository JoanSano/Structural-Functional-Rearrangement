import os
import logging
import yaml
from tqdm import tqdm
from subprocess import Popen, STDOUT, PIPE
from multiprocessing import Process
import re

from utils.steps import register_to_reference

def register_lesion_mask_Glioma(mask, config):
    """
    Register lesion mask to reference image
    """
    # Get the subject
    session = re.search(r'ses-.*/', mask).group().split("/")[0]
    subject = 'sub-'+re.search(r'PAT.[0-9]', re.search(r'ses-.*/', mask).group().split("/")[1]).group()
    # Get reference image
    ref_path = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session + '/anat/'
    output = Popen(f"find {ref_path} -maxdepth 1 -name '*.nii*'", shell=True, stdout=PIPE).stdout.read()
    ref_img = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    output = Popen(f"find {ref_path} -name '*.txt'", shell=True, stdout=PIPE).stdout.read()
    ref_mat = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    # Register to reference
    mask_reg, _ = register_to_reference(
        input=mask, output=ref_path+subject+'_'+session+'_T1w_tumor', reference=ref_img[0], skip=config["data"]["skip"], matrix=ref_mat[0],
    )
    ref_postop = config["data"]["output_path"] + 'derivatives/' + subject + '/ses-postop/anat/'+subject+'_ses-postop_T1w.nii.gz'
    mask_postop = config["data"]["output_path"] + 'derivatives/' + subject + '/ses-postop/anat/'+subject+'_ses-postop_T1w_tumor.nii.gz'
    if os.path.exists(ref_postop):
        os.system(f"cp {mask_reg} {mask_postop}")
    return mask_reg

def register_lesion_mask_Stroke(mask, config):
    """
    Register lesion mask to reference image
    """
    # Get the subject
    session_acute = 'ses-acute'
    session_followup = 'ses-followup'
    session_followup_2 = 'ses-followup-2'
    subject = mask.split("/")[-1].split("_")[1]
    if subject[0] == '0':
        subject = 'sub-PAT'+subject[1:]
    else:
        subject = 'sub-PAT'+subject     
    logging.info(f" {subject} Lesion registration")     
    # Get reference image
    ref_acute = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_acute + '/anat/' + subject + '_' + session_acute + '_T1w.nii.gz'
    ref_followup = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_followup + '/anat/' + subject + '_' + session_followup + '_T1w.nii.gz'
    ref_followup_2 = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_followup_2 + '/anat/' + subject + '_' + session_followup_2 + '_T1w.nii.gz'
    # Temporary output  
    tmp_acute = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_acute + '/anat/' + subject + '_' + session_acute + '_T1w_stroke-tmp.nii.gz'
    tmp_followup = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_followup + '/anat/' + subject + '_' + session_followup + '_T1w_stroke-tmp.nii.gz'
    tmp_followup_2 = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_followup_2 + '/anat/' + subject + '_' + session_followup_2 + '_T1w_stroke-tmp.nii.gz'
    # Final output  
    final_acute = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_acute + '/anat/' + subject + '_' + session_acute + '_T1w_stroke.nii.gz'
    final_followup = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_followup + '/anat/' + subject + '_' + session_followup + '_T1w_stroke.nii.gz'
    final_followup_2 = config["data"]["output_path"] + 'derivatives/' + subject + '/' + session_followup_2 + '/anat/' + subject + '_' + session_followup_2 + '_T1w_stroke.nii.gz'
    
    # Masks are already in MNI, but they have different orientations, dimensions and affines
    is_there = False
    if os.path.exists(ref_acute):
        os.system(f"cp {mask} {tmp_acute}")
        os.system(f"fslswapdim {tmp_acute} -x y z {tmp_acute} && fslorient -swaporient {tmp_acute}")
        os.system(f"mrgrid {tmp_acute} regrid {final_acute} -template {ref_acute} -force -quiet")
        os.system(f"rm {tmp_acute}")
        is_there = True
    if os.path.exists(ref_followup):
        os.system(f"cp {mask} {tmp_followup}")
        os.system(f"fslswapdim {tmp_followup} -x y z {tmp_followup} && fslorient -swaporient {tmp_followup}")
        os.system(f"mrgrid {tmp_followup} regrid {final_followup} -template {ref_followup} -force -quiet")
        os.system(f"rm {tmp_followup}")
        is_there = True
    if os.path.exists(ref_followup_2):
        os.system(f"cp {mask} {tmp_followup_2}")
        os.system(f"fslswapdim {tmp_followup_2} -x y z {tmp_followup_2} && fslorient -swaporient {tmp_followup_2}")
        os.system(f"mrgrid {tmp_followup_2} regrid {final_followup_2} -template {ref_followup_2} -force -quiet")
        os.system(f"rm {tmp_followup_2}")
        is_there = True
    if not is_there:
        logging.info(f" {subject} No lesion mask")

if __name__ == '__main__':
    ####### Preliminaires #######
    logging.basicConfig(level=logging.INFO)
    with open('config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    ####### Get masks of the desired subjects ######
    logging.info(f" Getting masks from {config['data']['input_path']}masks/")
    if 'Glioma' in config['data']['input_path']:
        output = Popen(f"find {config['data']['input_path']}masks/ -wholename '*/BCB_DM/*ras.nii*'", shell=True, stdout=PIPE).stdout.read()
    else:
        output = Popen(f"find {config['data']['input_path']}masks/ -wholename '*.nii*'", shell=True, stdout=PIPE).stdout.read()
    lesion_masks = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    Nmasks = len(lesion_masks)
    
    ####### Set number of threads #######
    if int(config["process"]["threads"]) >= len(os.sched_getaffinity(0)):
        # Use the available number of cores
        num_threads = len(os.sched_getaffinity(0))
    else:
        # Use the specified number of cores
        num_threads = int(config["process"]["threads"])

    if config["process"]["type"] == 'multi':
        ### Multiprocessing the subjects ###
        procs = []
        for i in tqdm(range(Nmasks)):
            lesion = lesion_masks[i]
            if 'Glioma' in config['data']['input_path']:
                p = Process(target=register_lesion_mask_Glioma, args=(lesion, config))
            else:
                p = Process(target=register_lesion_mask_Stroke, args=(lesion, config))
            p.start()
            procs.append(p)

            while len(procs)%num_threads == 0 and len(procs) > 0:
                for p in procs:
                    # wait for 10 seconds to wait process termination
                    p.join(timeout=10)
                    # when a process is done, remove it from processes queue
                    if not p.is_alive():
                        procs.remove(p)
                        
            # Final chunk could be shorter than num_cores, so it's handled waiting for its completion 
            #       (join without arguments wait for the end of the process)
            if i == Nmasks - 1:
                for p in procs:
                    p.join()
    else:
        ### Sequential processing of the subjects ###
        for i in tqdm(range(Nmasks)):
            lesion = lesion_masks[i]
            if 'Glioma' in config['data']['input_path']:
                register_lesion_mask_Glioma(lesion, config)
            else:
                register_lesion_mask_Stroke(lesion, config)