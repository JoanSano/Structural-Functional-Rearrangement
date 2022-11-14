import os
from subprocess import Popen, STDOUT, PIPE
import logging
import re

def check_path(path):
    """
    Check if specified path exist. Make directory in case it doesn't.
    """
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
    return path

def get_subjects(data_path, session='*', subject_ID='*', format='.csv', exclude =''):
    """
    Get all the subjects specified in directory.
    """

    logging.info(" Target Dataset: " + data_path)
    output = Popen(f"find {data_path if len(data_path)>0 else '.'} ! -wholename '{exclude}' -wholename '*{session}/*{subject_ID}*{format}'", shell=True, stdout=PIPE)
    files= str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    logging.info(" Found " + str(len(files)) + " subject(s)")
    return files

def get_info(f):
    """
    Get all the relevant info from the file.
    """
    
    f = f.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    path = f.removesuffix(f.split('/')[-1])
    name = (f.split('/')[-1]).split('.')[0] # just the file name (without extension)
    subject_ID = str(re.findall("sub-...[0-9][0-9]", name)[0])
    try:
        session = str(re.findall("_ses-p.+p", name)[0]) 
    except:
        session = ''
    return path, session, subject_ID, name

if __name__ == '__main__':
    pass