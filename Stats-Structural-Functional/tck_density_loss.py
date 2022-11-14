import pandas as pd
import numpy as np

from utils.paths import get_subjects, get_info

if __name__ == '__main__':
    # Loading tckmaps for healthy preops
    # TODO: Wait for finish msmtcsd

    # Loading tckmpas and lesions for patients
    preop_tckmaps_hybrid = get_subjects('./datasets/structural/images/', session='ses-preop', subject_ID='*', format='tckmap.nii.gz', exclude ='')
    lesions = []
    for f in preop_tckmaps_hybrid:
        _, _, subject_ID, name = get_info(f)
        lesions.append(get_subjects('./datasets/structural/images/', session='ses-preop', subject_ID=subject_ID, format='tumor.nii.gz', exclude ='')[0])

    