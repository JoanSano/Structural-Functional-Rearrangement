import logging
import os

from utils.paths import check_path

def fmri_steps(output_directory, config, subject, img_tag="latest"):
    ##################################
    # fMRIprep preprocessing steps ###
    ##################################
    # REFERENCES:
    # 1) https://fmriprep.org/en/stable/usage.html
    # 2) https://andysbrainbook.readthedocs.io/en/latest/OpenScience/OS/fMRIPrep_Demo_2_RunningAnalysis.html
    # I recommend using single processing in the config file and adding a high number of threads

    # If you want to process a single or list of subjects then use --participant-label flag in the commands.
    nthreads = config['process']['threads']
    fmriprep_output = output_directory + "fmriprep/"
    WD_fmriprep = "./fmriprep-workdir" 
    check_path(WD_fmriprep) 
    if config['data']['skip'] and os.path.exists(fmriprep_output+subject+'.html'):
        logging.info(f" {subject} fMRIprep summary available ... skipping")
    else:
        logging.info(f" Doing fMRIprep on {subject}")
        cmd = f"fmriprep-docker {config['data']['input_path']} {fmriprep_output} -w {WD_fmriprep} --use-aroma \
            --fs-license {config['fmri']['FS_license']} --fs-no-reconall --use-syn-sdc  \
            --nthreads {config['process']['threads']} --participant-label {subject} --nthreads {nthreads} > {WD_fmriprep}/{subject}.txt"
        os.system(cmd) #--output-space MNI152Lin --output-spaces MNI152NLin2009cAsym:res-2 
        logging.info(f" {subject} fMRIprep finished!")
    
    ###############################
    # XCP-D preprocessing steps ###
    ###############################
    # REFERENCES:
    # 1) https://xcp-d.readthedocs.io/en/latest/index.html
    NS_regressors = config['fmri']['NS_REG']
    WD_xcpd = "xcp-d_workdir"
    check_path(WD_xcpd)
    logging.info(f" Doing XCP-D steps on {subject}")
    cmd = f"docker run --rm -it -v {fmriprep_output}:/data/ -v {WD_xcpd}:/wkdir -v {output_directory}:/out  \
        pennlinc/xcp_d:{img_tag} /data /out -w /wkdir -p {NS_regressors} --participant-label {subject} --nthreads {nthreads} > {WD_xcpd}/{subject}.txt"
    os.system(cmd)
    logging.info(f" {subject} XCP-D steps finished!")