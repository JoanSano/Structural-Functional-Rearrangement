import pandas as pd
import numpy as np
from subprocess import Popen, STDOUT, PIPE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import nibabel as nib

from brainspace.gradient import GradientMaps
from brainspace.datasets import load_conte69, load_parcellation
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres

def plot_gradients(bold_directory, output_directory, parcellation, DAS_file, rois):
    Num_components = 10
    Num_gradients = 2
    schot = True # Set to false to render the image in interactive mode instead of saving
    threshold = 0.1
    sparcity = 0.9
    kernel = 'pearson'
    approach = 'dm'

    surf_lh, surf_rh = load_conte69() 
    labeling = load_parcellation('schaefer', scale=rois, join=True)
    mask = labeling != 0

    ############################################################
    # HCP

    HCP_mat = pd.read_csv(
        "/home/hippo/.local/lib/python3.10/site-packages/brainspace/datasets/matrices/main_group/schaefer_"+str(rois)+"_mean_connectivity_matrix.csv",
        sep=",",header=None).values
    HCP_mat = np.where(np.abs(HCP_mat)>=threshold, HCP_mat, 0)
    #HCP_mat = np.where(np.abs(HCP_mat)>=threshold, 1, 0)
    plotting.plot_matrix(HCP_mat, figure=(10, 8),# labels=labels[1:],
                        vmax=1, vmin=-1,
                        reorder=False)
    plt.savefig(output_directory+"HCP-FC.png", dpi=900)
    plt.close()

    g_HCP = GradientMaps(n_components=Num_components, random_state=0,
                        approach=approach, kernel=kernel)
    g_HCP.fit(HCP_mat, sparsity=sparcity)

    ############################################################
    # Connectivity Matrices

    output = Popen(f"find {bold_directory} -name *CON*residual.nii.gz", shell=True, stdout=PIPE)
    files_CON = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    output = Popen(f"find {bold_directory}ses-preop/ -name *PAT*residual.nii.gz", shell=True, stdout=PIPE)
    files_PAT = str(output.stdout.read()).removeprefix('b\'').removesuffix('\'').removesuffix('\\n').split('\\n')
    atlas_file = nib.load(parcellation)

    maps = np.array(atlas_file.get_fdata(),dtype=np.int16)
    labels = np.unique(maps)
    masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=True)
    correlation_measure = ConnectivityMeasure(kind='correlation')

    conn_matrix = np.zeros((rois,rois))
    pat_matrix = []
    subject_list = ["Healthy"]
    
    for f in files_CON:
        time_series = masker.fit_transform(f)
        mat = correlation_measure.fit_transform([time_series])[0]
        #mat = np.abs(correlation_measure.fit_transform([time_series])[0])
        conn_matrix += np.where(np.abs(mat)>=threshold, mat, 0)/len(files_CON)
        #conn_matrix += np.where(mat>=threshold, mat, 0)/len(files_CON)

    for i,f in enumerate(files_PAT):
        subject_list.append(f.split("/")[-1][0:9])
        time_series = masker.fit_transform(f)
        mat = correlation_measure.fit_transform([time_series])[0]
        #mat = np.abs(correlation_measure.fit_transform([time_series])[0])
        pat_matrix.append(np.where(np.abs(mat)>=threshold, mat, 0))
        #pat_matrix.append(np.where(mat>=threshold, mat, 0))

    fig, ax = plt.subplots(figsize=(10,7))
    ax.remove()
    plt.gcf().text(0.03, 0.75, "A", fontsize=20, fontweight="bold")
    plt.gcf().text(0.175, 0.75, "HEALTHY", fontsize=25, fontweight="bold")
    plt.gcf().text(0.523, 0.85, "TUMOR PATIENTS", fontsize=25, fontweight="bold")

    left, bottom, width, height = [0.01, 0.2, 0.5, 0.5]
    inset = fig.add_axes([left, bottom, width, height])
    aa = inset.imshow(conn_matrix, cmap=plt.get_cmap('coolwarm'), vmin=-1, vmax=1)
    inset.set_xticks([]), inset.set_xticklabels([])
    inset.set_yticks([]), inset.set_yticklabels([])
    cbar = fig.colorbar(aa, ax=inset, location='left', pad=0.02)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['-1', '1'])
    cbar.set_label("Correlation", fontsize=15, labelpad=0)

    for i,m in enumerate(pat_matrix):
        x,y = 0.5-i*0.1/len(pat_matrix), 0.3-i*0.2/len(pat_matrix)
        left, bottom, width, height = [x, y, 0.5, 0.5]
        inset = fig.add_axes([left, bottom, width, height])
        inset.imshow(m, cmap=plt.get_cmap('coolwarm'), vmin=-1, vmax=1)
        inset.set_xticks([]), inset.set_xticklabels([])
        inset.set_yticks([]), inset.set_yticklabels([])
    
    plt.savefig(output_directory+"Matrices.png", dpi=900)
    plt.close()

    ############################################################
    # Compute gradients

    gm = GradientMaps(n_components=Num_components, random_state=0,
                        approach=approach, kernel=kernel, alignment='procrustes')
    gm.fit(conn_matrix, sparsity=sparcity, reference=g_HCP.gradients_)

    gpat = GradientMaps(n_components=1, random_state=0,
                        approach=approach, kernel=kernel, alignment='procrustes')
    gpat.fit(pat_matrix, sparsity=sparcity, reference=np.expand_dims(gm.gradients_[:,0],axis=1))

    gradients = np.concatenate(
        (np.expand_dims(gm.gradients_[:,0],axis=1), np.array(gpat.gradients_).T[0]),
        axis=1)

    # First 2 gradients healthy cohorts
    grad = [None] * Num_gradients
    for i,g in enumerate(gm.gradients_.T):
        if i<Num_gradients: 
            grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)
        else:
            break

    cmap_name_gd1_2 = 'brg' 
    cmap_name_gd1 = 'viridis_r'
    # Cluster each region (point in surface) corresponds to gradient splitting
    from sklearn.cluster import KMeans
    grad_array = np.array(grad)
    grad_array = grad_array[:,~np.isnan(grad_array).any(axis=0)] # Discard nan values
    gd1_pred = KMeans(n_clusters=2, random_state=0).fit_predict(grad_array.T) 
    gd1_2_pred = KMeans(n_clusters=3, random_state=0).fit_predict(grad_array.T)
    gd1_plot, gd1_2_plot = [], []
    i = 0
    for g in grad[0]:
        if np.isnan(g):
            gd1_plot.append(np.nan)
            gd1_2_plot.append(np.nan)
        else:
            gd1_plot.append(gd1_pred[i])
            gd1_2_plot.append(gd1_2_pred[i])
            i += 1
    gd1_plot, gd1_2_plot = np.array(gd1_plot), np.array(gd1_2_plot)  
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(left=0.08,
                    bottom=0.08, 
                    right=0.82, 
                    top=0.82)
    cm = plt.cm.get_cmap(cmap_name_gd1_2)
    sc_gds = plt.scatter(grad_array[1,:], grad_array[0,:], c=gd1_2_pred, cmap=cm, s=2)
    #sc_gds = plt.scatter(grad[1], grad[0], c=grad[0], cmap=cm)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xlabel("Gradient 2"), ax.set_ylabel("Gradient 1")
    ax.set_xticks([-30,-15,0,15,30]), ax.set_xticklabels(["-30","-15","0","15","30"])
    ax.set_yticks([-30,0,30,60]), ax.set_yticklabels(["-30","0","30","60"])
    ax.set_xlim([-33,38]), ax.set_ylim([-40,62])
    #fig.colorbar(sc_gds, pad=0.01)

    left, bottom, width, height = [0.08, 0.84, 0.74, 0.14]
    inset = fig.add_axes([left, bottom, width, height])
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False), inset.spines['left'].set_visible(False)
    inset.set_xticks([]), inset.set_xticklabels([])
    inset.set_yticks([]), inset.set_yticklabels([])
    inset.set_xlim([-33,38])
    inset.hist(grad[1], bins=25, density=True, color='black', alpha=.6)

    left, bottom, width, height = [0.84, 0.08, 0.14, 0.74]
    inset = fig.add_axes([left, bottom, width, height])
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False), inset.spines['bottom'].set_visible(False)
    inset.set_xticks([]), inset.set_xticklabels([])
    inset.set_yticks([]), inset.set_yticklabels([])
    inset.set_ylim([-40,62])
    inset.hist(np.where(gd1_plot==1, grad[0], np.nan),
        bins=25, density=True, orientation='horizontal', color='indigo')
    inset.hist(np.where(gd1_plot==0, grad[0], np.nan),
        bins=25, density=True, orientation='horizontal', color='gold')

    variance = gm.lambdas_*100/np.sum(gm.lambdas_)
    left, bottom, width, height = [0.85, 0.85, 0.11, 0.11]
    inset = fig.add_axes([left, bottom, width, height])
    inset.scatter(range(gm.lambdas_.size), variance, s=2.5, c='black')
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_xticks([]), inset.set_xticklabels([])#, inset.yaxis.set_ticks_position('none')
    inset.set_yticks([variance[0]]), inset.set_yticklabels([str(int(variance[0]))])
    inset.set_xlabel("Gradient #"), inset.set_title("% Variance")

    plt.savefig(output_directory+"Gradients_1-2.png", dpi=900)
    plt.close() 
  
    plot_hemispheres(surf_lh, surf_rh, array_name=gd1_plot,
        color_bar=False, size=(900, 600), cmap=cmap_name_gd1, zoom=1.4, transparent_bg=False,
        filename=output_directory+"/Clusters_G1.png", screenshot=schot, interactive=False, layout_style='grid')
    plot_hemispheres(surf_lh, surf_rh, array_name=gd1_2_plot, 
        color_bar=False, size=(900, 600), cmap=cmap_name_gd1_2, zoom=1.4, transparent_bg=False,
        filename=output_directory+"/Clusters_G1vsG2.png", screenshot=schot, interactive=False, layout_style='grid')

    ############################################################
    # 1st Gradient correlations between regions (Healthy vs Patients)""" 
    
    correlations = np.zeros((len(subject_list)-1,))
    for i in range(1, len(subject_list)):
        correlations[i-1],_ = np.abs(pearsonr(gradients[:,0], gradients[:,i]))
    sorted_correlations = np.sort(correlations)

    DAS = pd.read_csv(DAS_file, sep='\t')
    DAS.set_index(DAS.Subject, inplace=True)
    DAS.drop(['Subject'], axis=1, inplace=True)
    mean_DAS = np.array([dict(DAS["DAS"])[k] for k in subject_list[1:]])
    mean_abs_DAS = np.abs(mean_DAS)

    pat_list = subject_list[1:]
    sorted_pat_list = [pat_list[i] for i in np.argsort(correlations)]
    fig, ax = plt.subplots(figsize=(6,8))
    plt.subplots_adjust(left=0.1,
                    bottom=0.12, 
                    right=0.98, 
                    top=0.98)
    plt.gcf().text(0.02, 0.96, "B", fontsize=20, fontweight="bold")
    ax.plot(sorted_correlations, 'ok')
    ax.set_xticks(range(0,len(files_PAT)))
    ax.set_xticklabels(sorted_pat_list, rotation='vertical')
    ax.set_ylabel("PCC 1st Gradient")
    ax.legend(frameon=False), ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    plt.savefig(output_directory+"Sorted-Correlations.png", dpi=900)
    plt.close()

    ############################################################
    # Is there anything in here?

    info = pd.read_csv('./datasets/participants.tsv', sep='\t')
    info = info[info["participant_id"].str.contains("CON") == False]
    info.set_index(info.participant_id, inplace=True)
    info.drop(['participant_id'], axis=1, inplace=True)
    info.index.name = None
    tumor_sizes = np.array([dict(info["tumor size (cub cm)"])[k] for k in subject_list[1:]])
    tumor_types = np.array([1 if 'ningioma' in dict(info["tumor type & grade"])[k] else 2 for k in subject_list[1:]])
    tumor_locs = np.array([1 if 'Frontal' in dict(info["tumor location"])[k] else 2 for k in subject_list[1:]])
    tumor_grades = np.array([2 if 'II' in dict(info["tumor type & grade"])[k] else 1 for k in subject_list[1:]])
    tumor_ventricles = np.array([2 if 'yes' in dict(info["ventricles"])[k] else 1 for k in subject_list[1:]])
    
    print("PCC vs DAS")
    print(pearsonr(correlations, mean_DAS))
    print(pearsonr(correlations, mean_abs_DAS))
    print("PCC vs Type")
    print(pearsonr(correlations, tumor_types))
    print("PCC vs Size")
    print(pearsonr(correlations, tumor_sizes))
    print("PCC vs Ventricle")
    print(pearsonr(correlations, tumor_ventricles))
    print("PCC vs Location")
    print(pearsonr(correlations, tumor_locs))
    print("PCC vs Grade")
    print(pearsonr(correlations, tumor_grades))

    ############################################################
    # Plot 1st Gradients

    """ grad = [None] * gradients.shape[-1]
    to_plot = 5

    for i,g in enumerate(gradients.T):
        grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)

    plot_hemispheres(surf_lh, surf_rh, array_name=grad[:3], 
        color_bar=True, size=(1300, 600), cmap='viridis_r', zoom=1.2,
        filename="./Gradients.png", screenshot=schot, transparent_bg=False) """

