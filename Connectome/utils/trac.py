import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
import os

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
    FODs should be mutually exclusive since a simple addition is done.
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