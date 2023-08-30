import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
import os
import glob
import numpy as np

def tck2trk(anatomical, trac_file):
    """
    Converts .tck files to .trk using an anatomical reference image. 
        Credits to nibabel: https://github.com/nipy/nibabel/blob/master/nibabel/cmdline/tck2trk.py
    """
    trac_format = nib.streamlines.detect_format(trac_file)
    if trac_format is not nib.streamlines.TckFile:
        raise ValueError("Tractogram loaded is not of .tck format")
    else:
        anat = nib.load(anatomical)
        converted, _ = os.path.splitext(trac_file)
        converted = converted + '.trk'
        
        if not os.path.exists(converted):
            # Build header using infos from the anatomical image.
            header = {}
            header[Field.VOXEL_TO_RASMM] = anat.affine.copy()
            header[Field.VOXEL_SIZES] = anat.header.get_zooms()[:3]
            header[Field.DIMENSIONS] = anat.shape[:3]
            header[Field.VOXEL_ORDER] = "".join(aff2axcodes(anat.affine))

            tck = nib.streamlines.load(trac_file)
            nib.streamlines.save(tck.tractogram, converted, header=header)

def trk2tck():
    #TODO: Very similar, but not needed at this moment
    pass

def lesion_deletion(main_mask, lesion_mask, directory):
    """
    Combines a brain binary mask and substract the pathological tissue from it (lesion_mask).
    Returns the name of the final mask.
    """
    # Loading masks
    img = nib.load(main_mask)
    lesion = nib.load(lesion_mask)

    # Substract the lesion and clip to positive values
    hole = img.get_fdata() - lesion.get_fdata()
    hole = hole.clip(min=0)

    # Saving
    name = 'NO-LESION_' + main_mask.split('/')[-1]
    full_name = directory + name
    nib.save(nib.Nifti1Image(hole, img.affine, img.header), full_name)
    return full_name

def upsample(input, output, factor=1.5):
    """
    MRtrix3 upsampmling image to increase/decrease resolution.
    """
    os.system( 
        f"mrgrid {input} regrid {output} -force -quiet -voxel {factor}"
    )
    return output

def merge_fods(fod1, fod2, directory):
    """
    Merges FODs from two different files. 
    FODs need to be mutually exclusive since a simple addition is done.
    Possible future options could include some kind of averaging, but 
        not sure it's super useful.
    Inputs shoudl be in .nii.gz format.
    """
    wm1 = nib.load(fod1)
    wm2 = nib.load(fod2)
    wm = wm1.get_fdata() + wm2.get_fdata()
    
    full_name = directory + 'FODs'
    nib.save(nib.Nifti1Image(wm, wm1.affine, wm1.header), full_name+'.nii.gz')
    os.system(f"mrconvert {full_name}.nii.gz {full_name}.mif -quiet -force")
    os.system(f"rm {full_name}.nii.gz {fod1} {fod2}")
    return full_name+'.mif'

def extract_tissue_response_shell(response, config_shell):
    with open(response, 'r') as f:
        full_response = f.readlines()
    
    new_response = ["# Shells: 0,"+config_shell['b_val']+"\n"]
    new_response.append(full_response[1])
    new_response.append(full_response[2])
    new_response.append(full_response[1+int(config_shell['num'])])
    return new_response
    
def extract_response_shell(response_1, new_response_1,
                           response_2, new_response_2,
                           response_3, new_response_3,
                           config_shell
                        ):
    """
    Extracts the response function corresponding to a particular shell from a multitissue 
        repsonse function.
    """
    with open(new_response_1, 'w') as f:
        f.writelines(extract_tissue_response_shell(response_1, config_shell))
    with open(new_response_2, 'w') as f:
        f.writelines(extract_tissue_response_shell(response_2, config_shell))
    with open(new_response_3, 'w') as f:
        f.writelines(extract_tissue_response_shell(response_3, config_shell))

def extract_healthy_streamlines_from_lesion(healthy_dir, lesion, output_file, subjec_ID):
    """
    Inputs
    --------
    healthy_dir: Directory where the healthy pool is located. The tractograms need be
        named according to BIDS format (e.g., sub-CONXX_ses-YY_extra-arguments.tck) 
        together with the .tck format extension
    lesion: NIFTI image containing the lesion mask. Needles to say that it needs to be
        corregistered with the same template as the healthy cohort
    subject_ID: ID of the patient. Only for summary purposes on the output file
    output_file: txt file containing the number of streamlines emerging a given lesion. The
        structure of the data is as follows:
            # Patient:\tsub-PATXX
            # Healthy-subjects:\tYY
            # --------------------
            sub-CON01_ses-preop:\t350909
            ...
            sub-CONZZ_ses-RRRRR:\t675648
            Lesion:\tTHE AVERAGE OF ALL PREVIOUS ENTRIES
            
    Returns
    --------
    returns the average number of streamlines that emerge from the lesion with respect to the
        healthy pool of subjects
    """

    tract_files = glob.glob(f"{healthy_dir}/**/*.tck", recursive=True)
    lines = [
        f"# Patient:\t{subjec_ID}\n",
        f"# Healthy-subjects:\t{len(tract_files)}\n",
        f"# ---------------------"
    ]
    np.random.seed(int(subjec_ID[-2:]))
    file_ID = "./ignore_"+str(np.random.randint(0,50))+"-"+str(np.random.randint(0,50))+"-"+str(np.random.randint(0,50))+"-"+str(np.random.randint(0,50))
    average = 0
    f = open(output_file, 'w')
    f.writelines(lines)
    for i,tract in enumerate(tract_files):
        ID = "_".join(tract.split("/")[-1].split("_")[:-1])
        f.writelines(f"\n{ID}:\t")
        os.system(f"tckedit {tract} {file_ID}.tck -include {lesion} -quiet -force")
        os.system(f"tckstats -output count {file_ID}.tck -quiet > {file_ID}.txt")
        with open(f"{file_ID}.txt", 'r') as g:
            streams = g.readlines()[0]
        f.writelines(f"{streams}")
        average = int(streams) + average
    average = average/len(tract_files)
    f.writelines(f"\nAverage-number:\t{average}")
    f.close()
    os.system(f"rm {file_ID}.tck {file_ID}.txt")
    return round(average)

def read_healthy_streamlines_from_lesion(streamline_file):
    with open(streamline_file,'r') as f:
        average = float(f.readlines()[-1].split("\t")[-1])
    return round(average)

if __name__ == '__main__':
    # Convert single tck file
    import argparse
    parser = argparse.ArgumentParser(description='Convert .tck to .trk')
    parser.add_argument('--anat', type=str, help='Anatomical image')
    parser.add_argument('--tck', type=str, help='Tractogram file .tck')
    parser.add_argument('--type', type=str, default='tck', help='Type of the input file')
    args = parser.parse_args()

    if args.type == 'tck':
        tck2trk(args.anat, args.tck)
    else:
        pass # Add a trk -> tck converter