import os
import logging
import re
from tqdm import tqdm
from multiprocessing import Process

from utils.paths import get_subjects
from methods.multishells import connectome_msmt_csd
from methods.singleshells import connectome_ss3t_csd
from methods.hybrids import hybrid

def run_connectome(config, acronym):
    """
    Generate a connectome for all the subjects specified in the config file.
    """

    ### DWI images to process ###
    files = get_subjects(config)

    ### Checking threads ###
    if int(config["threads"]) >= len(os.sched_getaffinity(0)):
        # Use the available number of threads
        num_threads = len(os.sched_getaffinity(0))
    else:
        # Use the specified number of threads
        num_threads = int(config["threads"])

    if config['process'] == 'multi' and len(files)>1:
        ### Multiprocessing the subjects ###
        procs = []
        for i in tqdm(range(len(files))):
            if config['recons'] == ('MSMT-CSD' or '' or None):
                p = Process(target=connectome_msmt_csd, args=(config, files[i], acronym))
            elif config['recons'] == 'SS3T-CSD':
                p = Process(target=connectome_ss3t_csd, args=(config, files[i], acronym))
            elif config['recons'] == 'hybrid':
                p = Process(target=hybrid, args=(config, files[i], acronym))
            else:
                raise ValueError("Please Provide a valid reconstruction method")
            p.start()
            procs.append(p)

            while len(procs)%num_threads == 0 and len(procs) > 0:
                for p in procs:
                    # wait for 10 seconds to wait process termination
                    p.join(timeout=10)
                    # when a process is done, remove it from processes queue
                    if not p.is_alive():
                        procs.remove(p)
                        
            # Final chunk could be shorter than num_threads, so it's handled waiting for its completion 
            #       (join without arguments wait for the end of the process)
            if i == len(files) - 1:
                for p in procs:
                    p.join()
    else:
        ### Sequential processing of the subjects ###
        for i in tqdm(range(len(files))):
            if config['recons'] == ('MSMT-CSD' or '' or None):
                connectome_msmt_csd(config, files[i], acronym)
            elif config['recons'] == 'SS3T-CSD':
                connectome_ss3t_csd(config, files[i], acronym)
            elif config['recons'] == 'hybrid':
                hybrid(config, files[i], acronym)
            else:
                raise ValueError("Please Provide a valid reconstruction method")

if __name__ == '__main__':
    pass
