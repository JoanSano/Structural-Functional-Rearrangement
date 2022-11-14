from ast import arg
import yaml
import logging
from tqdm import tqdm
from multiprocessing import Process
import argparse

from utils.paths import *
from runner import preprocess_subject
from fmri_preprocess import fmri_steps

parser = argparse.ArgumentParser(description='Preprocess a set of anat/dwi/fMRI subjects')
parser.add_argument('--config', type=str, default='config', help='Path to the config file')
args = parser.parse_args()

if __name__ == "__main__":
    ####### Preliminaires #######
    config_file = args.config
    if '.yaml' not in config_file:
        config_file = config_file + '.yaml'
    logging.basicConfig(level=logging.INFO)
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    if config["data"]["skip"]:
        logging.info(" Skip already available files option is set to 'True' ")
    else:
        logging.info(" Skip already available files option is set to 'False': Files will be overwritten ")

    ####### Generate derivatives directory #######
    output_directory = config["data"]["output_path"]+'derivatives/'
    check_path(output_directory)

    ####### Get subjects ######
    logging.info(f" Getting folders from {config['data']['input_path']}")
    subjects, Nsubjects = get_folders(config=config, search_type='directory', exclude='*derivatives*')
    logging.info("=====================================================================================")
    logging.info(
        f" PreProcessing {', '.join(config['subjects']['mris'])} images from {Nsubjects} subject(s) in {', '.join(config['subjects']['sessions'])} sessions(s)..."
    )
    logging.info("=====================================================================================")
    if "func" in config["subjects"]["mris"]:
        print("The fmri preprocessing pipeline uses fMRIprep and XCP-D workflows.")
        print("Input data must be in BIDS format, otherwise there is no garantee that the pipeline will work appropiately.")
        print("You need:\n \
            1) docker \
            2) fmriprep-docker installed via pip install \
            3) A FreeSurfer license file")
        print("Logs of both pipelines will be saved in the working directories as SUBJECT_ID_logs.txt and xcp-d_logs.txt.")
        logging.info("=====================================================================================")
    #    fmri_steps(output_directory, config)
    #else:
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
        for i in tqdm(range(Nsubjects)):
            subject = subjects[i]
            p = Process(target=preprocess_subject, args=(subject, output_directory, config))
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
            if i == Nsubjects - 1:
                for p in procs:
                    p.join()
    else:
        ### Sequential processing of the subjects ###
        for i in tqdm(range(Nsubjects)):
            subject = subjects[i]
            preprocess_subject(subject, output_directory, config)
