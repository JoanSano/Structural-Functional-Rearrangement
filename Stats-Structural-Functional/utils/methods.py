import pandas as pd
import numpy as np
import nibabel as nib
import os
from scipy import ndimage
from scipy.stats import pearsonr

def pcc(x, y):
    cc = 0
    for i in range(x.shape[0]):
        cc += np.corrcoef(x[i,:],y[i,:])[0,1]/x.shape[0]
    return cc

def mse(x, y):
    mse = 0
    for i in range(x.shape[0]):
        mse += np.mean((x[i,:] - y[i,:])**2,axis=0)/x.shape[0]
    return mse

def ApproximateEntropy(U, m=2, r=1) -> float:
    """Approximate_entropy."""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return _phi(m) - _phi(m + 1)

def complexity(Net, bins=10):
    """
    Calculates the complexity of a network.
    """
    flat_net = Net[np.triu_indices(Net.shape[0], k = 1)]
    hist, _ = np.histogram(flat_net, bins=bins)
    hist = hist/np.sum(hist)
    cmpl = 0
    for i in range(bins):
        cmpl += np.abs( hist[i] - 1/bins)
    cmpl = 1 - cmpl * (bins/(2*(bins-1)))
    return cmpl
    
def FFT_sampling_points (delta_t, length=180, bin1=2.4, bin2=2.1, eps=0.0001):
    """
    Calculates the number of sampling points for the FFT based on the
        current fmri series. For some reason the (idiot) MR technitian
        decided to use a temproal bin of either 2.4 or 2.1 seconds.
    To be used only in the rfft function!

    Inputs:
    -------

    Outputs:
    --------

    """
    bin = np.max([bin1, bin2])
    if (delta_t >= bin-eps)*(delta_t <= bin+eps):
        extra_bins = 0
    else:
        extra_bins = int(np.abs(bin1-bin2)*length/delta_t)+1
        
    points = length + extra_bins
    if points%2 == 0:
        fft_length = int(length/2 + 1)
    else:
        fft_length = int((length+1)/2)
    
    return points, fft_length

def Power_Analysis_Oedema_vs_Healthy(ii, Nc, session, subject_niftis_path,
        CONTROL_paired, C_subjects_paired, CONTROL_unpaired, C_subjects_unpaired, 
        PATIENT_paired, P_subjects_paired, PATIENT_unpaired, P_subjects_unpaired, sum_name, tissue='tumor'):
    # Load patient data
    if ii >= len(P_subjects_paired):
        subject, file = P_subjects_unpaired[ii-len(P_subjects_paired)], PATIENT_unpaired[ii-len(P_subjects_paired)]  
        patient = nib.load(file)
    else:
        subject, files = P_subjects_paired[ii], PATIENT_paired[ii]         
        fi = files[0] if session == 'ses-preop' else files[1]
        patient = nib.load(fi)
    bold = patient.get_fdata()
    delta_t = patient.header["pixdim"][4]
    dim1, dim2, dim3, dim4 = bold.shape 
    time_axis_P = np.arange(0,dim4*delta_t,delta_t)
    results_patients = pd.DataFrame(columns=['Subject', 'Total power', 'Power per bin (%)', 'Cumulative Power (%)', 'Oedema BOLD signal', 'time (s)'])
    print(f"Analysing {subject} in {session}")

    lesion_name = f"../Data/structural/images/ses-preop/lesion_MNI_ses-preop_3mm/{subject}_ses-preop_T1w_{tissue}_3mm.nii.gz"
    try:
        # Load lesion
        lesion = nib.load(lesion_name)
        oedema = np.repeat(lesion.get_fdata()[:,:,:,np.newaxis], bold.shape[3], axis=3)
        oedema = np.where(oedema > 0.05, 1, 0)
        # Patient Voxel_wise mean-temporal BOLD
        masked_bold = bold * oedema
    except:
        name = f"../Data/structural/images/ses-preop/lesion_MNI_ses-preop_3mm/regridded_tumor.nii.gz"
        os.system(f"mrgrid {lesion_name} regrid {name} -size {dim1},{dim2},{dim3} -force -quiet")# Load lesion
        lesion = nib.load(name)
        oedema = np.repeat(lesion.get_fdata()[:,:,:,np.newaxis], bold.shape[3], axis=3)
        oedema = np.where(oedema > 0.05, 1, 0)
        # Patient Voxel_wise mean-temporal BOLD
        masked_bold = bold * oedema
        os.system(f"rm {name}") 
        print(f"Regrided lesion mask of {subject} in {session}")
    masked_mean = np.mean(masked_bold, axis=3)
    nib.save(nib.Nifti1Image(masked_mean, patient.affine, patient.header), subject_niftis_path+subject+f"_{session}_voxel-wise_mean-temporal.nii.gz")

    # Whole-oedema temporal BOLD
    nan_bold = np.where(oedema >= 0.05, masked_bold, np.nan)
    masked_bold_voxel = nan_bold.reshape((nan_bold.shape[0]*nan_bold.shape[1]*nan_bold.shape[2], nan_bold.shape[-1]))
    mean_patient = np.nanmean(masked_bold_voxel, axis=0)

    # Fourier reconstruction
    P_error_fft_reconstruction, cutoffs = np.zeros((10,)), np.zeros((10,), dtype=np.int64)
    bin_P_power, P_power = np.zeros((10,)), np.zeros((10,))
    n_points, fft_length = FFT_sampling_points(delta_t, length=len(mean_patient))
    fft = np.fft.rfft(mean_patient, n=n_points)
    fft = fft[:fft_length]
    lower_cut = 0
    for j, percent in enumerate(range(10, 110, 10)):
        cutoff = int(len(fft)*percent/100)
        cutoffs[j] = percent
        fft_cut = fft[:cutoff]
        bin_P_power[j] = np.sum(np.square(np.abs(fft[lower_cut:cutoff])))
        P_power[j] = np.sum(np.square(np.abs(fft_cut)))
        ifft_cut = np.fft.irfft(fft_cut, dim4)
        P_error_fft_reconstruction[j] = ((mean_patient - ifft_cut)**2).mean()
        lower_cut = cutoff
    Full_Power_P = np.sum(np.square(np.abs(np.fft.rfft(mean_patient)))) #P_power[-1]#
    bin_P_power = 100*bin_P_power/P_power[-1]
    P_power = 100*P_power/P_power[-1]
    results_patients.loc[len(results_patients.index)] = [subject, Full_Power_P, bin_P_power, P_power, mean_patient, time_axis_P]

    # Loading controls
    print("Starting Healthy analysis")
    mean_control, time_axis_H = np.zeros((Nc,dim4)), []
    H_error_fft_reconstruction = np.zeros((Nc,10))
    bin_H_power, H_power, Full_H_power = np.zeros((Nc,10)), np.zeros((Nc,10)), np.zeros((Nc,))
    for i in range(Nc):
        # preop
        if i >= len(C_subjects_paired):
            C_subject, C_file = C_subjects_unpaired[i-len(C_subjects_paired)], CONTROL_unpaired[i-len(C_subjects_paired)]  
            C_fi = C_file
        else:
            C_subject, C_files = C_subjects_paired[i], CONTROL_paired[i]        
            C_fi = C_files[0] if session == 'ses-preop' else C_files[1]
        try: 
            healthy = nib.load(C_fi)
            healthy_bold = healthy.get_fdata()
            masked_bold_healthy = healthy_bold * oedema
        except:
            name = f'../Data/functional/images/control/{session}/regridded.nii.gz'
            os.system(f"mrgrid {C_fi} regrid {name} -size {dim1},{dim2},{dim3} -force -quiet")
            healthy = nib.load(name)
            healthy_bold = healthy.get_fdata()    
            masked_bold_healthy = healthy_bold * oedema
            os.system(f"rm {name}") 
            print(f"Regridded {C_subject} in {session} with respect to {subject}")
        masked_mean_healthy = np.mean(masked_bold_healthy, axis=3)
        nib.save(nib.Nifti1Image(masked_mean_healthy, healthy.affine, healthy.header), subject_niftis_path+C_subject+f'_{session}_voxel-wise_mean-temporal.nii.gz')
        
        # Temporal dimension
        delta_t_H = healthy.header["pixdim"][4]
        time_axis_H.append(np.arange(0,bold.shape[-1]*delta_t_H,delta_t_H))

        nan_bold_control = np.where(oedema >= 0.05, masked_bold_healthy, np.nan)
        masked_bold_voxel_control = nan_bold_control.reshape((nan_bold_control.shape[0]*nan_bold_control.shape[1]*nan_bold_control.shape[2], nan_bold_control.shape[-1]))
        mean_control[i,...] = np.nanmean(masked_bold_voxel_control, axis=0)

        # Fourier reconstruction
        n_points, fft_length = FFT_sampling_points(delta_t_H, length=len(mean_control[i,...]))
        fft = np.fft.rfft(mean_control[i,...], n=n_points)
        fft = fft[:fft_length]
        lower_cut = 0
        for j, percent in enumerate(range(10, 110, 10)):
            cutoff = int(len(fft)*percent/100)
            fft_cut = fft[:cutoff]
            bin_H_power[i,j] = np.sum(np.square(np.abs(fft[lower_cut:cutoff])))
            H_power[i,j] = np.sum(np.square(np.abs(fft_cut)))
            ifft_cut = np.fft.irfft(fft_cut, dim4)
            H_error_fft_reconstruction[i,j] = ((mean_control[i,...] - ifft_cut)**2).mean()
            lower_cut = cutoff
        Full_H_power[i] = np.sum(np.square(np.abs(np.fft.rfft(mean_control[i,...])))) #H_power[i,-1] #
        bin_H_power[i,:] = 100*bin_H_power[i,:]/H_power[i,-1]
        H_power[i,:] = 100*H_power[i,:]/H_power[i,-1]
        # Writing results in file
        results_patients.loc[len(results_patients.index)] = [C_subject, Full_H_power[i], bin_H_power[i,:], H_power[i,:], mean_control[i,...], time_axis_H[-1]]
    results_patients.to_csv(sum_name, sep='\t', index=False)

def DMN_overlap(subject, session, mm='3', tissue='tumor'):
    # Overlap
    DMN_img = nib.load(f"../Data/atlas/DMN_{mm}{mm}{mm}mm.nii")
    DMN = DMN_img.get_fdata()
    DMN = DMN[...,0] if len(DMN.shape)==4 else DMN
    lesion_img = nib.load(f"../Data/structural/images/ses-preop/lesion_MNI_ses-preop_{mm}mm/{subject}_ses-preop_T1w_{tissue}_{mm}mm.nii.gz")
    lesion, affine = lesion_img.get_fdata(), lesion_img.affine
    overlap_image = np.where(DMN * lesion >= 0.01, 1, 0)
    overlap = np.sum(overlap_image)/np.sum(np.where(DMN >= 0.01, 1, 0))
    name = f'../RESULTS/figures/functional/{session}/{subject}/niftis/DMN_overlap.nii.gz'
    nib.save(nib.Nifti1Image(overlap_image, DMN_img.affine, DMN_img.header), name)

    # Euclidean distance
    cm_DMN = np.load('../Data/atlas/DMN_Centroids.npy') # In MNI space coordinates
    cm_lesion = np.append(ndimage.center_of_mass(lesion), 1) # In voxel space (i.e., computed as voxel coordinates)
    cm_lesion_MNI = np.tile((affine @ cm_lesion)[:-1], (cm_DMN.shape[0],1)) # From voxel to MNI space - repeat for each DMN region
    E_distance = np.linalg.norm(cm_lesion_MNI-cm_DMN, axis=1) # Euclidean distance
    return overlap, E_distance.mean()
    
def BOLD_DMN(subject, session, path, mm='3', type_subject='healthy'):
    # DMN labels
    dmn = nib.load(f"../Data/atlas/DMN_{mm}{mm}{mm}mm.nii")
    dmn_data = dmn.get_fdata()
    dmn_labels = np.unique(dmn_data[np.nonzero(dmn_data)])

    # Healthy BOLD
    if type_subject == 'healthy':
        healthy = nib.load(f"../Data/functional/images/control/{session}/{subject}_{session}_task-rest_bold_residual.nii.gz")
        bold = healthy.get_fdata()
    elif type_subject == 'patient':
        healthy = nib.load(f"../Data/functional/images/{session}/{subject}_{session}_task-rest_bold_residual.nii.gz")
        bold = healthy.get_fdata()
    else:
        pass
    delta_t = healthy.header["pixdim"][4]
    dim1, dim2, dim3, dim4 = bold.shape 
    time_axis_P = np.arange(0,dim4*delta_t,delta_t)

    # BOLD signals of DMN 
    dmn_region_bold = np.zeros((len(dmn_labels), bold.shape[3]))
    signals = pd.DataFrame(columns=["Region", "Signal", "time (s)", 'Total power', 'Power per bin (%)', 'Cumulative Power (%)'])
    for i, region in enumerate(dmn_labels):
        mask_region = np.where(dmn_data == region, 1, 0)
        mask_region = np.repeat(mask_region[:,:,:,np.newaxis], bold.shape[3], axis=3)
        masked_bold_dmn = bold * mask_region
        nan_bold_dmn_region = np.where(mask_region >= 0.05, masked_bold_dmn, np.nan)
        masked_bold_voxel_control = nan_bold_dmn_region.reshape(
            (nan_bold_dmn_region.shape[0]*nan_bold_dmn_region.shape[1]*nan_bold_dmn_region.shape[2], nan_bold_dmn_region.shape[-1])
        )
        dmn_region_bold[i,...] = np.nanmean(masked_bold_voxel_control, axis=0)

        # Fourier reconstruction
        Error_fft_reconstruction, cutoffs = np.zeros((10,)), np.zeros((10,), dtype=np.int64)
        bin_power, Power = np.zeros((10,)), np.zeros((10,))

        n_points, fft_length = FFT_sampling_points(delta_t, length=len(dmn_region_bold[i,...]))
        fft = np.fft.rfft(dmn_region_bold[i,...], n=n_points)
        fft = fft[:fft_length]
        lower_cut = 0
        for j, percent in enumerate(range(10, 110, 10)):
            cutoff = int(len(fft)*percent/100)
            cutoffs[j] = percent
            fft_cut = fft[:cutoff]
            bin_power[j] = np.sum(np.square(np.abs(fft[lower_cut:cutoff])))
            Power[j] = np.sum(np.square(np.abs(fft_cut)))
            ifft_cut = np.fft.irfft(fft_cut, dim4)
            Error_fft_reconstruction[j] = ((dmn_region_bold[i,...] - ifft_cut)**2).mean()
            lower_cut = cutoff
        Full_Power = np.sum(np.square(np.abs(np.fft.rfft(dmn_region_bold[i,...])))) #Power[-1]#
        bin_power = 100*bin_power/Power[-1]
        Power = 100*Power/Power[-1]

        signals.loc[len(signals.index)] = [region, dmn_region_bold[i,...], time_axis_P, Full_Power, bin_power, Power]
    signals.to_csv(path+f"{subject}_{session}_DMN-region_BOLD.tsv", sep='\t', index=False)

def community_BOLD(data):
    """
    Returns the two communities based on the network built with pairwise correlations
    """
    from scipy.stats import pearsonr
    import networkx as nx
    import networkx.algorithms.community as nx_comm
    
    regions = data.shape[0]
    correlations = np.zeros((regions,regions))
    for r1 in range(regions):
        for r2 in range(r1+1, regions):
            correlations[r1,r2] = pearsonr(data[r1,:], data[r2,:])[0]
            correlations[r2,r1] = correlations[r1,r2]

    G = nx.from_numpy_array(correlations)
    return nx_comm.louvain_communities(G, seed=123), correlations

def read_DMN_summary(sum_name):
    # Read tsv summary
    dmn_bold = pd.read_csv(sum_name, sep='\t')
    dmn_bold.reset_index(drop=True, inplace=True)
    dmn_bold.set_index(dmn_bold.Region, inplace=True)

    # Define arrays
    dmn_region_bold = np.zeros((len(dmn_bold["Region"]), 180))
    dmn_region_time = np.zeros((len(dmn_bold["Region"]), 180))
    dmn_region_power_T = np.zeros((len(dmn_bold["Region"]),))
    dmn_region_power_cum = np.zeros((len(dmn_bold["Region"]),10))
    dmn_region_power_bin = np.zeros((len(dmn_bold["Region"]),10))

    # Load data
    for r, region in enumerate(dmn_bold["Region"]):
        dmn_region_bold[r,:] = np.array([float(k) for k in dmn_bold.loc[region, 'Signal'][1:-1].split()], dtype=np.float64)
        dmn_region_time[r,:] = np.array([float(k) for k in dmn_bold.loc[region, 'time (s)'][1:-1].split()], dtype=np.float64)
        dmn_region_power_T[r] = dmn_bold.loc[region, 'Total power']
        dmn_region_power_cum[r,:] = np.array([float(k) for k in dmn_bold.loc[region, 'Cumulative Power (%)'][1:-1].split()], dtype=np.float64)
        dmn_region_power_bin[r,:] = np.array([float(k) for k in dmn_bold.loc[region, 'Power per bin (%)'][1:-1].split()], dtype=np.float64)
    communities, corrnet = community_BOLD(dmn_region_bold)

    return dmn_region_bold, dmn_region_time, dmn_region_power_T, dmn_region_power_cum, dmn_region_power_bin, communities, dmn_bold["Region"].values, corrnet

def read_Odema_summary_healthy(sum_name):
    # Read tsv summary
    results_patient = pd.read_csv(sum_name, sep='\t')
    results_patient.reset_index(drop=True, inplace=True)
    results_patient.set_index(results_patient.Subject, inplace=True)

    # Define arrays
    active_healthy_pre = [k for k in results_patient['Subject'] if 'CON' in k]
    bold = np.zeros((len(active_healthy_pre), 180))
    time = np.zeros((len(active_healthy_pre), 180))
    power_T = np.zeros((len(active_healthy_pre),))
    power_cum = np.zeros((len(active_healthy_pre),10))
    power_bin = np.zeros((len(active_healthy_pre),10))

    # Load data
    for i, sub in enumerate(active_healthy_pre):
        bold[i,:] = np.array([float(k) for k in results_patient.loc[sub, 'Oedema BOLD signal'][1:-1].split()], dtype=np.float64)
        time[i,:] = np.array([float(k) for k in results_patient.loc[sub, 'time (s)'][1:-1].split()], dtype=np.float64)
        power_T[i] = results_patient.loc[sub, 'Total power']
        power_cum[i,:] = np.array([float(k) for k in results_patient.loc[sub, 'Cumulative Power (%)'][1:-1].split()], dtype=np.float64)
        power_bin[i,:] = np.array([float(k) for k in results_patient.loc[sub, 'Power per bin (%)'][1:-1].split()], dtype=np.float64)
    
    return bold, time, power_T, power_cum, power_bin

def read_Oedema_summary_patient(sum_name, subject):
    # Read tsv summary
    results_patient = pd.read_csv(sum_name, sep='\t')
    results_patient.reset_index(drop=True, inplace=True)
    results_patient.set_index(results_patient.Subject, inplace=True)

    # Load data
    bold = np.array([float(k) for k in results_patient.loc[subject, 'Oedema BOLD signal'][1:-1].split()], dtype=np.float64)
    time = np.array([float(k) for k in results_patient.loc[subject, 'time (s)'][1:-1].split()], dtype=np.float64)
    power_T = results_patient.loc[subject, 'Total power']
    power_cum = np.array([float(k) for k in results_patient.loc[subject, 'Cumulative Power (%)'][1:-1].split()], dtype=np.float64)
    power_bin = np.array([float(k) for k in results_patient.loc[subject, 'Power per bin (%)'][1:-1].split()], dtype=np.float64)

    return bold, time, power_T, power_cum, power_bin

def coef_pvals(X, Y, regressor):
    import scipy.stats as stats

    # Regression coefficients
    params = np.append(regressor.intercept_,regressor.coef_)
    predictions = regressor.predict(X)

    newX = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((Y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    names = ['Intercept', 'Type', 'Size', 'Location', 'Grade']#, 'Ventricular'

    myDF3 = pd.DataFrame()
    myDF3["Names"],myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [names,params,sd_b,ts_b,p_values]
 
    return myDF3, regressor.score(X, Y)

def permutation_test_correlation(x, y, num_permutations=1000):
    observed_corr, _ = pearsonr(x, y)
    permuted_correlations = []

    for _ in range(num_permutations):
        permuted_y = np.random.permutation(y)
        permuted_corr, _ = pearsonr(x, permuted_y)
        permuted_correlations.append(permuted_corr)

    p_value = (np.abs(permuted_correlations) >= np.abs(observed_corr)).mean()
    return observed_corr, p_value

if __name__ == '__main__':
    pass