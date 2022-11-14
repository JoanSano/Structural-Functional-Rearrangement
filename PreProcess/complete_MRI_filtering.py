import os
from subprocess import Popen, STDOUT, PIPE
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path to the dataset')
args = parser.parse_args()

def check_path(path):
    if not os.path.isdir(path):
        os.system("mkdir -p {}".format(path))
        return True
    else: 
        return False

if __name__ == '__main__':

    initial_dataset = args.dataset #"/home/bam/Joan/Stroke_WUSM_Dwi_CoRegis/derivatives/"
    """ final_dataset = "/home/bam/Joan/Stroke_Final/derivatives/"
    check_path(final_dataset) """

    report = open(args.dataset+"Filtering_report.txt", "w")

    output = Popen(f"find {initial_dataset} -type d -name 'sub-*'", shell=True, stdout=PIPE).stdout.read()
    subjects = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

    for subject in subjects:
        output = Popen(f"find {subject} -type d -name 'ses-*'", shell=True, stdout=PIPE).stdout.read()
        sessions = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
        for session in sessions:
            # Files to exist if not to be filtered
            anat = session + '/anat/' + subject.split("/")[-1] + '_' + session.split("/")[-1] + '_T1w.nii.gz'
            dwiAP = session + '/dwi/' + subject.split("/")[-1] + '_' + session.split("/")[-1] + '_acq-AP_dwi.nii.gz'
            dwiLR = session + '/dwi/' + subject.split("/")[-1] + '_' + session.split("/")[-1] + '_acq-LR_dwi.nii.gz'

            if os.path.exists(anat) and (os.path.exists(dwiAP) or os.path.exists(dwiLR)):
                """ target = final_dataset + subject.split("/")[-1] + '/' #+ session.split("/")[-1]
                check_path(target)
                os.system(f"cp -R {session} {target}") """
                pass
            else:
                report.write("Removed: " + session + "\n")
                os.system(f"rm -rf {session}")
        
        # Check whether the subject has any session left
        output = Popen(f"find {subject} -type d -name 'ses-*'", shell=True, stdout=PIPE).stdout.read()
        sessions = str(output).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
        if len(sessions[0]) == 0 :
            os.system(f"rm -rf {subject}")

    report.close()