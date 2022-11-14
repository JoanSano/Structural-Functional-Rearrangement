#!/bin/bash

# PreProcess anatomical
python3 main.py --config config_anat.yaml

# PreProcess structural
python3 main.py --config config_dwi.yaml

# Filter complete anat+dwi sessions and subjects
python3 complete_MRI_filtering.py --dataset /home/bam/Joan/Stroke_WUSM_Dwi_CoRegis/derivatives/