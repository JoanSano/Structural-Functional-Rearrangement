import json
import os
from subprocess import Popen, STDOUT, PIPE

def find_secondary_acquisitions(whole_path, subjectID, session, acq, weight):
    """
    Returns all the secondary acquisitions in the directory.
    """
    # All possible encodings
    directions = ['acq-AP', 'acq-PA', 'acq-LR', 'acq-RL', 'acq-IS', 'acq-SI']
    files = []
    for dir in directions:
        possible_file = whole_path+subjectID+'_'+session+'_'+dir+'_'+weight
        if dir != acq:
            if os.path.exists(possible_file+'.nii'):
                files.append([possible_file+'.nii.gz', possible_file+'.bval', possible_file+'.bvec', possible_file+'.json'])
            elif os.path.exists(possible_file+'.nii.gz'):
                files.append([possible_file+'.nii.gz', possible_file+'.bval', possible_file+'.bvec', possible_file+'.json'])
            else:
                pass
    if len(files) == 0:
        available = False
    else:
        available = True
    return files, available 

def write_acquisition_line(json_data, directions):
    """
    Returns a single line of the acquisition parameters file for the given json data.
    """
    ## Vector encoding the acquisition direction ##
    aqp_dir = directions[json_data['PhaseEncodingDirection']]

    ## TotalReadouttime or an approximation of it ##
    try:
        t_readout = json_data["TotalReadoutTime"]
    except:
        try: 
            es_codename = "EffectiveEchoSpacing"
            if es_codename not in json_data.keys():
                es_codename = "Estimated"+es_codename
                if es_codename not in json_data.keys():
                    es_codename = "EchoTime"
            ees = json_data[es_codename]
            amPE = json_data["AcquisitionMatrixPE"]
            t_readout = round(ees * (amPE - 1), 4)
        except:
            t_readout = 0.043
            
    ## Linein the acquisition parameters file format ##
    return str(aqp_dir[0])+' '+str(aqp_dir[1])+' '+str(aqp_dir[2])+' '+str(t_readout)+'\n'
    

def acq_file(jsons, acqp_path):
    """ 
    Call this function to generate the acquisition paramaters'
    file used in both topup and eddy corrections.
    Information:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq#How_do_I_know_what_to_put_into_my_--acqp_file
    jsons: List containing the paths to the json files. 
           The order matters; the first json file is the first line.
    """

    assert type(jsons) == list    
    dir_to_vec = {'j': [0, 1, 0], 'j-': [0, -1, 0],
                  'i': [1, 0, 0], 'i-': [-1, 0, 0],
                  'k': [0, 0, 1], 'k-': [0, 0, -1]}

    acq_f = open(acqp_path, "w")
    Nacqs = len(jsons)
    for i in range(Nacqs):
        ## We write the line according to the acquisition ##
        json_data = json.load(open(jsons[i]))
        acq_f.write(write_acquisition_line(json_data, dir_to_vec))
    acq_f.close()


def index_file(files, index_path):
    """ 
    Call this function to generate the index file used in eddy.
    Information:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--acqp
    files: List containing the paths to the files.
    """

    ### BE SURE THAT YOU WRITE THIS FILE CORRECTLY ACCORDING TO YOUR DATA!! ###
    index = open(index_path, "w")
    for j, acq in enumerate(files):
        # Number of volumes for a given acquisition
        message = Popen(f"fslnvols {acq}", shell=True, stdout=PIPE).stdout.read()
        Nvolumes = int(str(message).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')[0])
        for i in range(Nvolumes):
            index.write(str(j+1)+' ')
    index.close()

if __name__=='__main__':
    pass

