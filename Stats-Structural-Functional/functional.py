import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.stats import ttest_1samp, pearsonr, spearmanr

from utils.data import prepare_functional_files
from utils.paths import check_path
from utils.methods import *
from utils.figures_functional import *
from gradients import *

parser = argparse.ArgumentParser(description='Analysis of BOLD signals in Oedemas from brain tumors')
parser.add_argument('--sessions', type=str, default='ses-preop', help='Session to analyze. * for all sessions. Input as comma separated list.')
parser.add_argument('--figures', type=str, default='false', help='Make figures of results')
args = parser.parse_args()

if __name__ == '__main__':
    figs = True if args.figures.lower()=='true' else False
    fig_fmt = "svg"
    ###################################
    ### Get files in an ordered way ###
    ###################################
    (CONTROL_paired, C_subjects_paired), (CONTROL_unpaired, C_subjects_unpaired) = prepare_functional_files("../Data/functional/images/", sessions='*', subject="CON")
    (PATIENT_paired, P_subjects_paired), (PATIENT_unpaired, P_subjects_unpaired) = prepare_functional_files("../Data/functional/images/", sessions='*', subject="PAT", exclude='*lesion*')
    
    ######################
    ### Tumor Features ###
    ######################
    info = pd.read_csv('Data/participants.tsv', sep='\t')
    info = info[info["participant_id"].str.contains("CON") == False]
    info.set_index(info.participant_id, inplace=True)
    info.drop(['participant_id'], axis=1, inplace=True)
    info.index.name = None
    tumor_sizes = np.array([dict(info["tumor size (cub cm)"])[k] for k in P_subjects_paired])
    tumor_types = np.array([1 if 'ningioma' in dict(info["tumor type & grade"])[k] else 2 for k in P_subjects_paired])
    tumor_locs = np.array([1 if 'Frontal' in dict(info["tumor location"])[k] else 2 for k in P_subjects_paired])
    tumor_grade = np.array([2 if 'II' in dict(info["tumor type & grade"])[k] else 1 for k in P_subjects_paired])
    tumor_ventricles = np.array([2 if 'yes' in dict(info["ventricles"])[k] else 1 for k in P_subjects_paired])
    tumor_sizes_unpaired = np.array([dict(info["tumor size (cub cm)"])[k] for k in P_subjects_unpaired])
    tumor_types_unpaired = np.array([1 if 'ningioma' in dict(info["tumor type & grade"])[k] else 2 for k in P_subjects_unpaired])
    tumor_locs_unpaired = np.array([1 if 'Frontal' in dict(info["tumor location"])[k] else 2 for k in P_subjects_unpaired])
    tumor_grade_unpaired = np.array([2 if 'II' in dict(info["tumor type & grade"])[k] else 1 for k in P_subjects_unpaired])
    tumor_ventricles_unpaired = np.array([2 if 'yes' in dict(info["ventricles"])[k] else 1 for k in P_subjects_unpaired])
    
    ################
    ### Analysis ###
    ################
    sessions = ['ses-preop','ses-postop'] if args.sessions=='*' else args.sessions.split(',')
    for session in sessions:
        Nc = len(C_subjects_paired)+len(C_subjects_unpaired) if session == 'ses-preop' else len(C_subjects_paired)
        Np = len(P_subjects_paired)+len(P_subjects_unpaired) if session == 'ses-preop' else len(P_subjects_paired)
            
        ######################
        ### Prepare arrays ###
        ######################
        if session == 'ses-preop':
            #Oedema
            DAS_Oedema_pre = pd.DataFrame(columns=['Subject', 'DAS', 'SEM'])  
            Total_Power_Patients_pre, Bin_Power_Patients_pre, Cum_Power_Patients_pre = np.zeros((Np,)), np.zeros((Np,10)), np.zeros((Np,10))
            Total_Power_Healthy_pre, Bin_Power_Healthy_pre, Cum_Power_Healthy_pre = np.zeros((Np,Nc,)), np.zeros((Np,Nc,10)), np.zeros((Np,Nc,10))
            BOLD_oedema_Patient_pre, BOLD_oedema_Healthy_pre = np.zeros((Np,180)), np.zeros((Np,Nc,180))
            time_series_Patient_pre, time_series_Healthy_pre = np.zeros((Np,180)), np.zeros((Np,Nc,180))
            Relative_dynamics_pre, Relative_power_pre = np.zeros((Np,Nc)), np.zeros((Np,Nc))
            
            # DMN and oedema overlap
            DMN_lesion_overlap_pre, DMN_lesion_distance_pre = np.zeros((Np,)), np.zeros((Np,))

            # DMN arrays
            DAS_DMN_pre = pd.DataFrame(columns=['Subject', 'DAS', 'SEM'])  
            DMN_region_bold_healthy_pre, DMN_region_bold_patient_pre = np.zeros((Nc,41,180)), np.zeros((Np,41,180))
            DMN_region_power_T_healthy_pre, DMN_region_power_T_patient_pre = np.zeros((Nc,41,)), np.zeros((Np,41,))
            DMN_region_power_cum_healthy_pre, DMN_region_power_cum_patient_pre = np.zeros((Nc,41,10)), np.zeros((Np,41,10))
            DMN_region_power_bin_healthy_pre, DMN_region_power_bin_patient_pre = np.zeros((Nc,41,10)), np.zeros((Np,41,10))
            communities_healthy_pre, communities_patient_pre = [], []
            DMN_regions_healthy_pre, DMN_regions_patient_pre = np.zeros((Nc,41)), np.zeros((Np,41))
            DMN_regions_healthy_ACF_pre, DMN_regions_patient_ACF_pre = np.zeros((Nc,41,180)), np.zeros((Np,41,180))
            DMN_regions_healthy_ACF_pval_pre, DMN_regions_patient_ACF_pval_pre = np.zeros((Nc,41,180)), np.zeros((Np,41,180))
            DMN_CorrNet_healthy_pre, DMN_CorrNet_patient_pre = np.zeros((Nc,41,41)), np.zeros((Np,41,41))
            DMN_Complexity_healthy_pre, DMN_Complexity_patient_pre = np.zeros((Nc,)), np.zeros((Np,))
            DMN_pcc_sim_pre, DMN_mse_sim_pre = np.zeros((Np,Nc)), np.zeros((Np,Nc))
            Oedema_T_power_change_pre, DMN_dynamic_change_pre = np.zeros((Np,Nc)), np.zeros((Np,Nc))
            DMN_Richness_change_pre = np.zeros((Np,Nc))

        else:
            # Oedema
            DAS_Oedema_post = pd.DataFrame(columns=['Subject', 'DAS', 'SEM'])  
            Total_Power_Patients_post, Bin_Power_Patients_post, Cum_Power_Patients_post = np.zeros((Np,)), np.zeros((Np,10)), np.zeros((Np,10))
            Total_Power_Healthy_post, Bin_Power_Healthy_post, Cum_Power_Healthy_post = np.zeros((Np,Nc,)), np.zeros((Np,Nc,10)), np.zeros((Np,Nc,10))
            BOLD_oedema_Patient_post, BOLD_oedema_Healthy_post = np.zeros((Np,180)), np.zeros((Np,Nc,180))
            time_series_Patient_post, time_series_Healthy_post = np.zeros((Np,180)), np.zeros((Np,Nc,180))
            Relative_dynamics_post, Relative_power_post = np.zeros((Np,Nc)), np.zeros((Np,Nc))

            # DMN and oedema overlap
            DMN_lesion_overlap_post, DMN_lesion_distance_post = np.zeros((Np,)), np.zeros((Np,))

            # DMN arrays 
            DAS_DMN_post = pd.DataFrame(columns=['Subject', 'DAS', 'SEM'])  
            DMN_region_bold_healthy_post, DMN_region_bold_patient_post = np.zeros((Nc,41,180)), np.zeros((Np,41,180))
            DMN_region_power_T_healthy_post, DMN_region_power_T_patient_post = np.zeros((Nc,41,)), np.zeros((Np,41,))
            DMN_region_power_cum_healthy_post, DMN_region_power_cum_patient_post = np.zeros((Nc,41,10)), np.zeros((Np,41,10))
            DMN_region_power_bin_healthy_post, DMN_region_power_bin_patient_post = np.zeros((Nc,41,10)), np.zeros((Np,41,10))
            communities_healthy_post, communities_patient_post = [], []
            DMN_regions_healthy_post, DMN_regions_patient_post = np.zeros((Nc,41)), np.zeros((Np,41))
            DMN_regions_healthy_ACF_post, DMN_regions_patient_ACF_post = np.zeros((Nc,41,180)), np.zeros((Np,41,180))
            DMN_regions_healthy_ACF_pval_post, DMN_regions_patient_ACF_pval_post = np.zeros((Nc,41,180)), np.zeros((Np,41,180))
            DMN_CorrNet_healthy_post, DMN_CorrNet_patient_post = np.zeros((Nc,41,41)), np.zeros((Np,41,41))
            DMN_Complexity_healthy_post, DMN_Complexity_patient_post = np.zeros((Nc,)), np.zeros((Np,))
            DMN_pcc_sim_post, DMN_mse_sim_post = np.zeros((Np,Nc)), np.zeros((Np,Nc))
            Oedema_T_power_change_post, DMN_dynamic_change_post = np.zeros((Np,Nc)), np.zeros((Np,Nc))
            DMN_Richness_change_post = np.zeros((Np,Nc))

        #######################################################
        ### BOLD signals in DMN regions in healthy subjects ###
        #######################################################
        for i in range(Nc):
            if i >= len(C_subjects_paired):
                C_subject = C_subjects_unpaired[i-len(C_subjects_paired)]
            else:
                C_subject = C_subjects_paired[i] 

            C_path = check_path(f"RESULTS/numerical/functional/control/{session}/{C_subject}/")
            DMN_healthy_name = C_path+f"{C_subject}_{session}_DMN-region_BOLD.tsv"
            if not os.path.exists(DMN_healthy_name):
                print(f"{C_subject} in {session} writing {DMN_healthy_name}")
                BOLD_DMN(C_subject, session, C_path, mm='3')

            if session == 'ses-preop':
                DMN_region_bold_healthy_pre[i], dmn_region_time, DMN_region_power_T_healthy_pre[i], DMN_region_power_cum_healthy_pre[i], \
                    DMN_region_power_bin_healthy_pre[i], cms, DMN_regions_healthy_pre[i], DMN_CorrNet_healthy_pre[i] = read_DMN_summary(DMN_healthy_name)
                communities_healthy_pre.append(cms)       
                DMN_Complexity_healthy_pre[i] = complexity(DMN_CorrNet_healthy_pre[i], bins=15)   

                if figs:
                    C_path = check_path(f"RESULTS/figures/functional/control/{session}/{C_subject}/")
                    png_name = C_path + f"{C_subject}_{session}_DMN-region-BOLD."+fig_fmt
                    DMN_regions_healthy_ACF_pre[i], DMN_regions_healthy_ACF_pval_pre[i] = BOLD_DMN_regions(DMN_region_bold_healthy_pre[i], dmn_region_time, DMN_regions_healthy_pre[i], png_name, cms)
                    print(f"{C_subject}_{session} DMN ready")

            else:
                DMN_region_bold_healthy_post[i], dmn_region_time, DMN_region_power_T_healthy_post[i], DMN_region_power_cum_healthy_post[i], \
                    DMN_region_power_bin_healthy_post[i], cms, DMN_regions_healthy_post[i], DMN_CorrNet_healthy_post[i] = read_DMN_summary(DMN_healthy_name)
                communities_healthy_post.append(cms)
                DMN_Complexity_healthy_post[i] = complexity(DMN_CorrNet_healthy_post[i], bins=15)  

                if figs:
                    C_path = check_path(f"RESULTS/figures/functional/control/{session}/{C_subject}/")
                    png_name = C_path + f"{C_subject}_{session}_DMN-region-BOLD."+fig_fmt
                    DMN_regions_healthy_ACF_post[i], DMN_regions_healthy_ACF_pval_post[i] = BOLD_DMN_regions(DMN_region_bold_healthy_post[i], dmn_region_time, DMN_regions_healthy_post[i], png_name, cms)
                    print(f"{C_subject}_{session} DMN ready")
            print(f"{C_subject} in {session} DMN BOLD signal done")
            
        ########################################################
        ### Analysis of BOLD signals per patient and session ###
        ########################################################
        for pat in range(Np):
            print("=================================")
            if pat >= len(P_subjects_paired):
                # This setting ensures that the unpaired subjects are at the end
                subject = P_subjects_unpaired[pat-len(P_subjects_paired)]
            else:
                subject = P_subjects_paired[pat]  

            ### PATIENT OUTPUT DIRECTORIES ###
            subject_results_path = check_path(f"RESULTS/numerical/functional/{session}/{subject}/")
            subject_figures_path = check_path(f"RESULTS/figures/functional/{session}/{subject}/figs/")
            subject_niftis_path = check_path(f"RESULTS/figures/functional/{session}/{subject}/niftis/")

            ### PATIENT Default Mode Network ###
            DMN_patient_name = subject_results_path+f'{subject}_{session}_DMN-region_BOLD.tsv'
            if not os.path.exists(DMN_patient_name):
                print(f"{subject} in {session} writing {DMN_patient_name}")
                BOLD_DMN(subject, session, subject_results_path, mm='3', type_subject='patient')
            print(f"{subject} in {session} DMN BOLD signal done")

            ### PATIENT OEDEMA POWER ANALYSIS ###
            Oedema_Patient_name = subject_results_path+f'{subject}_{session}_analysis-power_oedema-BOLD_summary.tsv'
            if not os.path.exists(Oedema_Patient_name):
                print(f"{subject} in {session} writing {Oedema_Patient_name}")
                Power_Analysis_Oedema_vs_Healthy(pat, Nc, session, subject_niftis_path,
                        CONTROL_paired, C_subjects_paired, CONTROL_unpaired, C_subjects_unpaired, 
                        PATIENT_paired, P_subjects_paired, PATIENT_unpaired, P_subjects_unpaired, Oedema_Patient_name)  
            print(f"{subject} in {session} Oedema signal done")

            ### LOADING AND PLOTTING RESULTS ###
            if session == 'ses-preop':
                # DMN
                DMN_region_bold_patient_pre[pat], dmn_region_time, DMN_region_power_T_patient_pre[pat], DMN_region_power_cum_patient_pre[pat], \
                    DMN_region_power_bin_patient_pre[pat], cms, DMN_regions_patient_pre[pat], DMN_CorrNet_patient_pre[pat] = read_DMN_summary(DMN_patient_name)
                communities_patient_pre.append(cms)        
                DMN_lesion_overlap_pre[pat], DMN_lesion_distance_pre[pat] = DMN_overlap(subject, session)     
                DMN_pcc_sim_pre[pat,:] = np.array([pcc(DMN_CorrNet_patient_pre[pat], DMN_CorrNet_healthy_pre[c]) for c in range(Nc)])
                DMN_mse_sim_pre[pat,:] = np.array([mse(DMN_CorrNet_patient_pre[pat], DMN_CorrNet_healthy_pre[c]) for c in range(Nc)])
                DMN_Complexity_patient_pre[pat] = complexity(DMN_CorrNet_patient_pre[pat], bins=15)
                
                # Oedema
                BOLD_oedema_Patient_pre[pat,:], time_series_Patient_pre[pat,:], Total_Power_Patients_pre[pat], \
                    Cum_Power_Patients_pre[pat,:], Bin_Power_Patients_pre[pat,:] = read_Oedema_summary_patient(Oedema_Patient_name, subject)
                BOLD_oedema_Healthy_pre[pat,:,:], time_series_Healthy_pre[pat,:,:], Total_Power_Healthy_pre[pat,:], \
                    Cum_Power_Healthy_pre[pat,:,:], Bin_Power_Healthy_pre[pat,:,:] = read_Odema_summary_healthy(Oedema_Patient_name)
                
                # Changes
                DMN_dynamic_change_pre[pat,:] = np.array([np.trapz(
                    np.mean(DMN_region_power_cum_patient_pre[pat],axis=0)-np.mean(DMN_region_power_cum_healthy_pre[c],axis=0)
                ) for c in range(Nc)])
                DMN_Richness_change_pre[pat,:] = DMN_Complexity_patient_pre[pat] - DMN_Complexity_healthy_pre
                Oedema_T_power_change_pre[pat,:] = Total_Power_Patients_pre[pat] - Total_Power_Healthy_pre[pat,:]
                Relative_dynamics_pre[pat,:] = np.array([np.trapz(
                    Cum_Power_Patients_pre[pat,:]-Cum_Power_Healthy_pre[pat,c,:]
                ) for c in range(Nc)])
                DAS_Oedema_pre.loc[len(DAS_Oedema_pre.index)] = [subject, np.mean(Relative_dynamics_pre[pat,:]), np.std(Relative_dynamics_pre[pat,:])/np.sqrt(Nc)]
                DAS_DMN_pre.loc[len(DAS_DMN_pre.index)] = [subject, np.mean(DMN_dynamic_change_pre[pat,:]), np.std(DMN_dynamic_change_pre[pat,:])/np.sqrt(Nc)]
                
                if figs:
                    # DMN
                    png_name = subject_figures_path + f"{subject}_{session}_DMN-region-BOLD."+fig_fmt
                    DMN_regions_patient_ACF_pre[pat], DMN_regions_patient_ACF_pval_pre[pat] = BOLD_DMN_regions(DMN_region_bold_patient_pre[pat], dmn_region_time, DMN_regions_patient_pre[pat], png_name, cms, fig_fmt=fig_fmt)
                    # DMN comparisson
                    DMN_patient_vs_healthy(
                        DMN_region_bold_healthy_pre, DMN_region_bold_patient_pre[pat],
                        DMN_region_power_T_healthy_pre, DMN_region_power_cum_healthy_pre, DMN_region_power_bin_healthy_pre, DMN_regions_healthy_pre, 
                        communities_healthy_pre, DMN_CorrNet_healthy_pre, DMN_lesion_overlap_pre[pat], time_series_Healthy_pre[pat],
                        DMN_region_power_T_patient_pre[pat], DMN_region_power_cum_patient_pre[pat], DMN_region_power_bin_patient_pre[pat], DMN_regions_patient_pre[pat], 
                        communities_patient_pre[pat], DMN_CorrNet_patient_pre[pat], dmn_region_time,
                        DMN_regions_healthy_ACF_pre, DMN_regions_patient_ACF_pre[pat],
                        DMN_regions_healthy_ACF_pval_pre, DMN_regions_patient_ACF_pval_pre[pat],
                        DMN_Complexity_healthy_pre, DMN_Complexity_patient_pre[pat],
                        Nc, subject, session, info, subject_figures_path, fig_fmt=fig_fmt
                    )
                    print(f"{subject}_{session} DMN ready")
                    # Oedema
                    Power_Analysis_Figures(
                        Cum_Power_Healthy_pre[pat], Bin_Power_Healthy_pre[pat], Total_Power_Healthy_pre[pat], 
                        Cum_Power_Patients_pre[pat], Bin_Power_Patients_pre[pat], Total_Power_Patients_pre[pat], 
                        BOLD_oedema_Healthy_pre[pat], BOLD_oedema_Patient_pre[pat], time_series_Healthy_pre[pat], time_series_Patient_pre[pat],
                        Nc, subject, session, info, subject_figures_path, fig_fmt=fig_fmt
                    )  
                    print(f"{subject}_{session} Oedema ready")
            else:
                # DMN
                DMN_region_bold_patient_post[pat], dmn_region_time, DMN_region_power_T_patient_post[pat], DMN_region_power_cum_patient_post[pat], \
                    DMN_region_power_bin_patient_post[pat], cms, DMN_regions_patient_post[pat], DMN_CorrNet_patient_post[pat] = read_DMN_summary(DMN_patient_name)
                communities_patient_post.append(cms)
                DMN_lesion_overlap_post[pat], DMN_lesion_distance_post[pat] = DMN_overlap(subject, session)
                DMN_pcc_sim_post[pat,:] = np.array([pcc(DMN_CorrNet_patient_post[pat], DMN_CorrNet_healthy_post[c]) for c in range(Nc)])
                DMN_mse_sim_post[pat,:] = np.array([mse(DMN_CorrNet_patient_post[pat], DMN_CorrNet_healthy_post[c]) for c in range(Nc)])
                DMN_Complexity_patient_post[pat] = complexity(DMN_CorrNet_patient_post[pat], bins=15)

                # Oedema
                BOLD_oedema_Patient_post[pat,:], time_series_Patient_post[pat,:], Total_Power_Patients_post[pat], \
                    Cum_Power_Patients_post[pat,:], Bin_Power_Patients_post[pat,:] = read_Oedema_summary_patient(Oedema_Patient_name, subject)
                BOLD_oedema_Healthy_post[pat,:,:], time_series_Healthy_post[pat,:,:], Total_Power_Healthy_post[pat,:], \
                    Cum_Power_Healthy_post[pat,:,:], Bin_Power_Healthy_post[pat,:,:] = read_Odema_summary_healthy(Oedema_Patient_name)
                
                # Changes
                DMN_dynamic_change_post[pat,:] = np.array([np.trapz(
                    np.mean(DMN_region_power_cum_patient_post[pat],axis=0)-np.mean(DMN_region_power_cum_healthy_post[c],axis=0)
                ) for c in range(Nc)])
                DMN_Richness_change_post[pat,:] = DMN_Complexity_patient_post[pat] - DMN_Complexity_healthy_post
                Oedema_T_power_change_post[pat,:] = Total_Power_Patients_post[pat] - Total_Power_Healthy_post[pat,:]
                Relative_dynamics_post[pat,:] = np.array([np.trapz(
                    Cum_Power_Patients_post[pat,:]-Cum_Power_Healthy_post[pat,c,:]
                ) for c in range(Nc)])
                DAS_Oedema_post.loc[len(DAS_Oedema_post.index)] = [subject, np.mean(Relative_dynamics_post[pat,:]), np.std(Relative_dynamics_post[pat,:])/np.sqrt(Nc)]
                DAS_DMN_post.loc[len(DAS_DMN_post.index)] = [subject, np.mean(DMN_dynamic_change_post[pat,:]), np.std(DMN_dynamic_change_post[pat,:])/np.sqrt(Nc)]

                if figs:
                    # DMN
                    png_name = subject_figures_path + f"{subject}_{session}_DMN-region-BOLD."+fig_fmt
                    DMN_regions_patient_ACF_post[pat], DMN_regions_patient_ACF_pval_post[pat] = BOLD_DMN_regions(DMN_region_bold_patient_post[pat], dmn_region_time, DMN_regions_patient_post[pat], png_name, cms, fig_fmt=fig_fmt)
                    # DMN comparisson
                    DMN_patient_vs_healthy(
                        DMN_region_bold_healthy_post, DMN_region_bold_patient_post[pat],
                        DMN_region_power_T_healthy_post, DMN_region_power_cum_healthy_post, DMN_region_power_bin_healthy_post, DMN_regions_healthy_post, 
                        communities_healthy_post, DMN_CorrNet_healthy_post, DMN_lesion_overlap_post[pat], time_series_Healthy_post[pat],
                        DMN_region_power_T_patient_post[pat], DMN_region_power_cum_patient_post[pat], DMN_region_power_bin_patient_post[pat], DMN_regions_patient_post[pat], 
                        communities_patient_post[pat], DMN_CorrNet_patient_post[pat], dmn_region_time,
                        DMN_regions_healthy_ACF_post, DMN_regions_patient_ACF_post[pat],
                        DMN_regions_healthy_ACF_pval_post, DMN_regions_patient_ACF_pval_post[pat],
                        DMN_Complexity_healthy_post, DMN_Complexity_patient_post[pat],
                        Nc, subject, session, info, subject_figures_path, fig_fmt=fig_fmt
                    )
                    print(f"{subject}_{session} DMN ready")
                    # Oedema
                    Power_Analysis_Figures(
                        Cum_Power_Healthy_post[pat], Bin_Power_Healthy_post[pat], Total_Power_Healthy_post[pat], 
                        Cum_Power_Patients_post[pat], Bin_Power_Patients_post[pat], Total_Power_Patients_post[pat], 
                        BOLD_oedema_Healthy_post[pat], BOLD_oedema_Patient_post[pat], time_series_Healthy_post[pat], time_series_Patient_post[pat],
                        Nc, subject, session, info, subject_figures_path, fig_fmt=fig_fmt
                    )
                    print(f"{subject}_{session} Oedema ready")
    try:
        DAS_Oedema_pre.to_csv("RESULTS/numerical/functional/ses-preop/DAS_Oedema_pre.tsv", sep='\t', index=False)
        DAS_DMN_pre.to_csv("RESULTS/numerical/functional/ses-preop/DAS_DMN_pre.tsv", sep='\t', index=False)
    except:
        pass
    try:
        DAS_Oedema_post.to_csv("RESULTS/numerical/functional/ses-postop/DAS_Oedema_post.tsv", sep='\t', index=False)
        DAS_DMN_post.to_csv("RESULTS/numerical/functional/ses-preop/DAS_DMN_post.tsv", sep='\t', index=False)
    except:
        pass

    ############################
    ### Group Level analysis ###
    ############################
    max_index_DAS = np.argmax(DMN_dynamic_change_pre.mean(axis=1))
    min_index_DAS = np.argmin(DMN_dynamic_change_pre.mean(axis=1))
    print("=================================")
    if max_index_DAS > len(P_subjects_paired):
        print(f"Subject with Higher DMN DAS: {P_subjects_unpaired[max_index_DAS-len(P_subjects_paired)]}; DMN DAS={DMN_dynamic_change_pre.mean(axis=1)[max_index_DAS]}")
    else:
        print(f"Subject with Higher DMN DAS: {P_subjects_paired[max_index_DAS]}; DMN DAS={DMN_dynamic_change_pre.mean(axis=1)[max_index_DAS]}")
    if min_index_DAS >= len(P_subjects_paired):
        print(f"Subject with Lower DMN DAS: {P_subjects_unpaired[min_index_DAS-len(P_subjects_paired)]}; DMN DAS={DMN_dynamic_change_pre.mean(axis=1)[min_index_DAS]}")
    else:
        print(f"Subject with Lower DMN DAS: {P_subjects_paired[min_index_DAS]}; DMN DAS={DMN_dynamic_change_pre.mean(axis=1)[min_index_DAS]}")
    print("=================================")
    _, pT = ttest_1samp(DMN_Richness_change_pre.mean(axis=1), 0, alternative='two-sided')
    _, pU = mannwhitneyu(DMN_Richness_change_pre.mean(axis=1), 0, alternative='two-sided')
    print(f"Richness change {DMN_Richness_change_pre.mean()} with pT={pT} and pU={pU} __ two-tailed")
    _, pT = ttest_1samp(np.abs(DMN_Richness_change_pre).mean(axis=1), 0, alternative='greater')
    _, pU = mannwhitneyu(np.abs(DMN_Richness_change_pre).mean(axis=1), 0, alternative='greater')
    print(f"Absolut Richness change {np.abs(DMN_Richness_change_pre).mean()} with pT={pT} and pU={pU} __ one-tailed")
    r_power_das, p_power_das = pearsonr(DMN_dynamic_change_pre.mean(axis=1), DMN_region_power_T_patient_pre.mean(axis=1))
    print(f"Correlation DAS with Power DMN: r={r_power_das} with p-value={p_power_das}")
    r_power_das, p_power_das = pearsonr(DMN_dynamic_change_pre.mean(axis=1), Oedema_T_power_change_pre.mean(axis=1))
    print(f"Correlation DAS with Power Oedema: r={r_power_das} with p-value={p_power_das}")
    
    group_analysis(
        DMN_dynamic_change_pre, DMN_pcc_sim_pre, DMN_lesion_overlap_pre,
        DMN_lesion_distance_pre, DMN_Richness_change_pre, Relative_dynamics_pre,
        Nc, Np, 'ses-preop',
        Oedema_T_power_change_pre, Total_Power_Healthy_pre,
        np.concatenate((tumor_types,tumor_types_unpaired)),
        np.concatenate((tumor_sizes,tumor_sizes_unpaired)), 
        np.concatenate((tumor_ventricles,tumor_ventricles_unpaired)),
        np.concatenate((tumor_locs,tumor_locs_unpaired)),
        np.concatenate((tumor_grade,tumor_grade_unpaired)), fig_fmt="svg"
    )
    '''
    group_analysis(
        DMN_dynamic_change_pre, DMN_pcc_sim_pre, DMN_lesion_overlap_pre,
        DMN_lesion_distance_pre, DMN_Richness_change_pre, Relative_dynamics_pre,
        Nc, Np, 'ses-preop',
        Oedema_T_power_change_pre, Total_Power_Healthy_pre,
        np.concatenate((tumor_types,tumor_types_unpaired)),
        np.concatenate((tumor_sizes,tumor_sizes_unpaired)), 
        np.concatenate((tumor_ventricles,tumor_ventricles_unpaired)),
        np.concatenate((tumor_locs,tumor_locs_unpaired)),
        np.concatenate((tumor_grade,tumor_grade_unpaired)), fig_fmt="png"
    )
    '''

    """ ############################
    ### Functional gradients ###
    ############################
    print("=================================")
    print("Computing Functional Gradients")
    g_rois, Nnets = 100, 17
    parcellation = "./datasets/atlas/Schaefer2018_"+str(g_rois)+"Parcels_"+str(Nnets)+"Networks_order_FSLMNI152_1mm.nii.gz"
    DAS_file = "RESULTS/numerical/functional/ses-preop/DAS_DMN_pre.tsv"
    # Find more parcellations here: 
    # https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI
    plot_gradients("./datasets/functional_2/images/", "RESULTS/figures/functional/", 
            parcellation, DAS_file, g_rois) """