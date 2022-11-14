import os
from subprocess import Popen, STDOUT, PIPE
from tqdm import tqdm

tck_files_dir = '/home/hippo/Joan/conn-msmt_09-June-22/co-registered/'
file_type = '_SIFT30.tck'
output = Popen(f"find {tck_files_dir} -name *{file_type}* ",shell=True, stdout=PIPE)
files = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')

for f in tqdm(files):
	subject = f.split('/')[6]
	session = f.split('/')[7]
	ref_out = Popen(f"find /home/hippo/Joan/Glioma_Gent_AAL3_CoRegis/derivatives/co-registered/{subject}/{session}/anat/ -name *{subject}_{session}_T1w.nii.gz ",shell=True, stdout=PIPE)
	ref = str(ref_out.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')[0]
	
	output_f = '/home/hippo/Joan/conn-msmt_09-June-22/tckmaps/'+f.split('/')[-1].split('.')[0]+'_tckmap.nii.gz'
	
	os.system(f"tckmap -template {ref} {f} {output_f} -quiet -force")
	print(f"{subject}_{session} DONE!")
