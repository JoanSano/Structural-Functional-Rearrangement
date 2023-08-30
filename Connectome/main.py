import yaml
import logging
import os
import shutil
from datetime import datetime
import argparse
from connectome import run_connectome
from utils.paths import check_path

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='', help='Output folder name. Default contains the date and time of the execution.')
parser.add_argument('--configuration', type=str, default=".", help='Path to the config file specifications.')
parser.add_argument('--overwrite_config', action="store_true", help="Overwrite configuration file.")
opts = parser.parse_args()

if __name__ == '__main__':
    ### Open and copy config file - Set the output directory ###
    config_file = opts.configuration + "/config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    x = datetime.now()
    if opts.folder == '':
        acronym = 'Connectome_' + config['recons'].lower() + '_' + x.strftime("%b-%d-%Y_%H-%M") +'/'
    else:
        acronym = opts.folder + '/'
    if config['paths']['output_dir'] is not None:
        common_output = check_path(config['paths']['output_dir'] + acronym)
    else:
        common_output = check_path(config['paths']['dataset_dir'] + acronym)

    if opts.overwrite_config:
        shutil.copyfile('config.yaml', common_output+'config.yaml')

    ### Load info ###
    logging.basicConfig(level=logging.INFO)
    logging.info(" This script crucially relies on FSL, ANTS and MRtrix 3.0 softwares! Please make sure they are available on your machine.")
    if config['skip']:
        logging.info(" You have selected to skip methods that generate already available files. Please be sure those files are the correct ones.")
        x = input("Do you want to proceed? ([Y]/n)")
    else:
        x = 'Y'

    ### Execute ###
    if x == 'n':
        os.system(f"rm -rf {common_output}")
        quit()
    else:
        os.system("clear")        
        run_connectome(config, acronym)
    logging.info("-------------------------- Processing Done --------------------------") 