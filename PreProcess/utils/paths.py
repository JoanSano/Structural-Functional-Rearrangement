import os
from subprocess import Popen, STDOUT, PIPE
from pathlib import Path

def check_path(path):
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
        return True
    else: 
        return False

def bids_tree(folder, mris=['anat', 'dwi']):
    paths = dict()
    for i in range(len(mris)):
        paths[mris[i]] = folder+mris[i]+'/'
        check_path(paths[mris[i]])
    return paths

def get_folders(search_type='directory', maxdepth=1, mindepth=1, **kwargs):
    # Folders or files (not both)
    searches = {'directory': 'd', 'file': 'f'}
    if search_type not in searches.keys():
        raise ValueError("Invalid search. Expected one of: %s" % searches.keys())

    # Use provided arguments
    try:
        config = kwargs['config']
        data_path, subjects = config["data"]["input_path"], f"*{config['subjects']['folders']}*"
    except:
        data_path = kwargs['path']
        try:
            subjects = kwargs['subjects']
        except:
            subjects = "*"
    try:
        exclude = kwargs['exclude']
    except:
        exclude = ""
    
    # Search
    output = Popen(
        f"find {data_path if len(data_path)>0 else '.'} -maxdepth {str(maxdepth)} -mindepth {str(mindepth)} -type {searches[search_type]} ! -name '{exclude}' -name '*{subjects}*'", 
        shell=True, stdout=PIPE
    ).stdout.read()
    folders = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    return folders, len(folders)

def extract_nii(gzip_file):
    """
    Extracts a gzip file.
    """
    os.system(f"gzip -d {gzip_file} -f")

def get_info_from_nifti(nifti):
    """
    Returns all the information
    """
    nifti = nifti.removeprefix("\"").removeprefix("\'").removesuffix("\"").replace('./', '')
    whole_path = nifti.removesuffix(nifti.split('/')[-1])
    subjectID = nifti.split('/')[-4]
    session = nifti.split('/')[-3]
    mri = nifti.split('/')[-2]
    name = nifti.split('/')[-1].split('.')[0]
    if mri == 'dwi':
        acq = name.split('_')[-2]
        weight = 'dwi'
    elif mri == 'anat':
        acq = None
        weight = 'T1w'
    else:
        acq = None
        weight = None
    exts = nifti.split('/')[-1].split('.')[1:]
    extension = ''
    for i in exts:
        extension += '.'+i
    if extension == '.gz':
        Path(whole_path+name+extension).rename(whole_path+name+'.nii'+extension)
        extension = '.nii.gz'
    if 'gz' in extension:
        extract_nii(whole_path+name+extension)
        extension = '.nii'
    return whole_path, subjectID, session, mri, name, acq, weight, extension

def proceed(session, mri, acq, config):
    """
    Returns a boolean allowing or not for preprocessing of the file.
    """
    if config["subjects"]["sessions"] == "*" and mri in config["subjects"]["mris"]:
        if mri == 'dwi':
            if acq in config["main_acqs"]:
                return True
            else: 
                return False
        else:
            return True
    else:
        if session in config["subjects"]["sessions"] and mri in config["subjects"]["mris"]:
            if mri == 'dwi':
                if acq in config["main_acqs"]:
                        return True
                else: 
                    return False
            else:
                return True
        else:
            return False


if __name__ == ' __main__':
    pass