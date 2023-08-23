import matplotlib.pylab as plt
import numpy as np
from scipy.stats import ttest_1samp, mannwhitneyu, ttest_ind
import statsmodels.api as sm
from utils.methods import permutation_test_correlation

import gc

gc.collect()

#torch.cuda.empty_cache()


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=3):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black', lw=0.5)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

def Power_Analysis_Figures(
        H_power, bin_H_power, Full_H_power, 
        P_power, bin_P_power, Full_Power_P, 
        mean_control, mean_patient, time_axis_H, time_axis_P,
        Nc, subject, session, info, subject_figures_path, fig_fmt="png", tissue='Whole Tumor'
    ):

    _, p = ttest_1samp(Full_H_power, Full_Power_P, alternative='two-sided')
    if p<=0.05 and p>=0.01:
        pp = '*' 
    elif p<=0.01:
        pp = '**'
    else:
        pass

    cutoffs = np.arange(10,110,10)
    mean_power_cum, sem_power_cum = np.mean(H_power, axis=0), np.std(H_power, axis=0)/np.sqrt(Nc)
    mean_power_bin, sem_power_bin = np.mean(bin_H_power, axis=0), np.std(bin_H_power, axis=0)/np.sqrt(Nc)
    mean_power_full, sem_power_full = np.mean(Full_H_power, axis=0), np.std(Full_H_power, axis=0)/np.sqrt(Nc)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    t1 = ax.plot(cutoffs, P_power, '-o', color='red', linewidth=2, label=subject)
    t2 = ax.errorbar(cutoffs, mean_power_cum, yerr=sem_power_cum, fmt='-o', color='blue', alpha=.6, linewidth=1, label='Healthy')
    t3 = ax.bar(cutoffs-2, bin_P_power, width=4, color=[1,0,0,0.4])
    t4 = ax.bar(cutoffs+2, mean_power_bin, width=4, color=[0,0,1,0.4])
    ax.set_xlabel("$\omega_c$ (%)", fontsize=10), ax.set_ylabel("$P_{\omega_c}$ (%)", fontsize=10)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_title(dict(info["tumor type & grade"])[subject]+' - '+dict(info["tumor location"])[subject], fontsize=10)
    ax.set_xticks([0,20,40,60,80,100]), ax.set_xticklabels(['0','20','40','60','80','100'])
    ax.set_yticks([0,20,40,60,80,100]), ax.set_yticklabels(['0','20','40','60','80','100'])
    plt.legend(frameon=False, fontsize=10, ncol=1, loc='upper left')

    left, bottom, width, height = [0.7, 0.15, 0.2, 0.2]
    inset = fig.add_axes([left, bottom, width, height])
    in1 = inset.bar(0, Full_Power_P/mean_power_full, color=[1,0,0,0.4])
    in2 = inset.bar(1, 1, yerr=sem_power_full/mean_power_full, color=[0,0,1,0.4], error_kw=dict(lw=1.5), ecolor='k', capsize=3, width=0.75, align='center')
    if p <=0.05:
        barplot_annotate_brackets(0, 1, p, [0,1],[1, Full_Power_P/mean_power_full], dh=0.25, barh=.01, fs=7.5)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_xticklabels([]), inset.set_ylabel("Relative $P_T$", fontsize=10) #inset.set_ylabel("$P_T=\sum_k |A_k|^2$", fontsize=10)
    inset.set_yticks([0,1]), inset.set_yticklabels(["0","1"])

    left, bottom, width, height = [0.6, 0.5, 0.3, 0.2]
    inset = fig.add_axes([left, bottom, width, height])
    i_t1 = inset.plot(time_axis_P, mean_patient, color='red', linewidth=1, label=subject)
    for c in range(Nc):
        inset.plot(time_axis_H[c], mean_control[c,:], color='blue', linewidth=0.35, alpha=0.3)
        inset.plot(time_axis_H[c], time_axis_H[c]*0, color='black', linewidth=1)
    inset.set_title("BOLD signal", fontsize=8), inset.set_xlabel("time (s)", fontsize=6)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.spines['bottom'].set_visible(False), inset.set_xlim([0, 450])
    inset.set_xticks(np.arange(0,450,100)), inset.tick_params(axis='both', which='both', length=0)

    plt.savefig(subject_figures_path+subject+f'_{session}_power-analysis_BOLD-signal_oedema.'+fig_fmt, dpi=1000)
    plt.close()

    # BOLD activity inside Oedema
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.99, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=1)
    plt.gcf().text(0.01, 0.925, "A", fontsize=20, fontweight="bold")
    t1 = ax.plot(mean_patient, color='red', linewidth=2, label=subject)
    for c in range(Nc):
        ax.plot(mean_control[c,:], color='black', linewidth=1, alpha=0.3)
    ax.set_ylabel("BOLD", fontsize=14), ax.set_xlabel("time ($\Delta t$)", fontsize=10)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_title(f"{tissue} Signal", fontsize=14)
    plt.savefig(subject_figures_path+subject+f'_{session}_BOLD-signal_oedema.'+fig_fmt, dpi=1000)
    plt.close()

def BOLD_DMN_regions(bold_signals, time_axis, labels, name, communities, fig_fmt="png"):
    N_regions = bold_signals.shape[0]
    if N_regions%2 != 0: 
        nrows = N_regions//2 + 1
    else:
        nrows = N_regions//2

    fig, ax = plt.subplots(nrows, 2, figsize=(17, 28))
    plt.subplots_adjust(left=0.04,
                    bottom=0.025, 
                    right=0.99, 
                    top=0.99, 
                    wspace=0.1, 
                    hspace=1)
    ax = ax.flat
    for r in range(N_regions):
        if r in communities[0]:
            color = 'red'
        else:
            color = 'blue'
        if np.max(np.abs(bold_signals[r,:]))>5 and np.max(np.abs(bold_signals[r,:]))<10:
            lw = 2
        elif np.max(np.abs(bold_signals[r,:]))>10:
            lw = 3
        else:
            lw = 1

        ax[r].plot(time_axis[r,:], bold_signals[r,:], color=color, linewidth=lw)
        ax[r].plot(time_axis[r,:], time_axis[r,:]*0, color='black', linewidth=1)
        ax[r].set_title(f"Region {int(labels[r])}", color=color, fontweight="bold")
        ax[r].spines['right'].set_visible(False), ax[r].spines['top'].set_visible(False)
        ax[r].spines['bottom'].set_visible(False)
        ax[r].set_ylabel("BOLD"), ax[r].set_xlabel("time (s)")
        ax[r].set_xlim([0, time_axis[r,-1]+5])
        ax[r].set_xticks(np.arange(0,450, 50))
        ax[r].tick_params(axis='both', which='both', length=0)
    # Don't show the last empty axis
    ax[-1].spines['right'].set_visible(False), ax[-1].spines['top'].set_visible(False)
    ax[-1].spines['bottom'].set_visible(False), ax[-1].spines['left'].set_visible(False)
    ax[-1].set_yticks([]), ax[-1].set_xticks([])
    plt.savefig(name, dpi=1000)
    plt.close()

    acf = np.zeros((bold_signals.shape))
    p_val = np.zeros((bold_signals.shape))
    fig, ax = plt.subplots(nrows, 2, figsize=(17, 28))
    plt.subplots_adjust(left=0.04,
                    bottom=0.025, 
                    right=0.99, 
                    top=0.99, 
                    wspace=0.1, 
                    hspace=1)
    ax = ax.flat
    for r in range(N_regions):
        acf[r,:], _, p_val[r,1:] = sm.tsa.acf(bold_signals[r,:], nlags = bold_signals.shape[1]-1, qstat=True)
        if r in communities[0]:
            color = 'red'
        else:
            color = 'blue'
        if np.max(np.abs(bold_signals[r,:]))>5 and np.max(np.abs(bold_signals[r,:]))<10:
            lw = 2
        elif np.max(np.abs(bold_signals[r,:]))>10:
            lw = 3
        else:
            lw = 1

        ax[r].plot(time_axis[r,:], acf[r,:], color=color, linewidth=lw)
        ax[r].set_title(f"Region {int(labels[r])}", color=color, fontweight="bold")
        ax[r].spines['right'].set_visible(False), ax[r].spines['top'].set_visible(False)
        ax[r].set_ylabel("BOLD"), ax[r].set_xlabel("time (s)")
        ax[r].set_xlim([0, time_axis[r,-1]+5])
        ax[r].set_xticks(np.arange(0,450, 50))
    # Don't show the last empty axis
    ax[-1].spines['right'].set_visible(False), ax[-1].spines['top'].set_visible(False)
    ax[-1].spines['bottom'].set_visible(False), ax[-1].spines['left'].set_visible(False)
    ax[-1].set_yticks([]), ax[-1].set_xticks([])
    plt.savefig(".."+name.split('.')[-2]+'-ACF.'+fig_fmt, dpi=1000)
    plt.close()

    return acf, p_val

def DMN_patient_vs_healthy(
    DMN_bold_healthy, DMN_bold_patient,
    DMN_power_T_healthy, DMN_power_cum_healthy, DMN_power_bin_healthy, DMN_regions_healthy, communities_healthy, DMN_CorrNet_healthy, overlap, time_axis_healthy,
    DMN_power_T_patient, DMN_power_cum_patient, DMN_power_bin_patient, DMN_regions_patient, communities_patient, DMN_CorrNet_patient, dmn_region_time_patient,
    DMN_ACF_healthy, DMN_ACF_patient, DMN_regions_healthy_ACF_pval, DMN_regions_patient_ACF_pval,
    DMN_Complexity_healthy, DMN_Complexity_patient,
    Nc, subject, session, info, subject_figures_path, fig_fmt="png"
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(left=0.05,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=1)
    cutoffs = np.arange(10,110,10)
    ax2.annotate('A', xy =(3.3, 1), xytext =(3, 1.8)) 
    ax2.remove()
    plt.gcf().text(0.01, 0.95, "B", fontsize=20, fontweight="bold")
    plt.gcf().text(0.48, 0.95, "C", fontsize=20, fontweight="bold")
    plt.gcf().text(0.48, 0.43, "D", fontsize=20, fontweight="bold")

    # Cumulative and bin power
    patient_cp = np.mean(DMN_power_cum_patient, axis=0)
    healthy_nets_cp = np.mean(DMN_power_cum_healthy, axis=1)
    healthy_cp, sem_cp = np.mean(healthy_nets_cp, axis=0), np.std(healthy_nets_cp, axis=0)/np.sqrt(Nc)

    patient_bp = np.mean(DMN_power_bin_patient, axis=0)
    healthy_nets_bp = np.mean(DMN_power_bin_healthy, axis=1)
    healthy_bp, sem_bp = np.mean(healthy_nets_bp, axis=0), np.std(healthy_nets_bp, axis=0)/np.sqrt(Nc)

    patient_tp = np.mean(DMN_power_T_patient, axis=0)
    healthy_nets_tp = np.mean(DMN_power_T_healthy, axis=1)
    healthy_tp, sem_tp = np.mean(healthy_nets_tp, axis=0), np.std(healthy_nets_tp, axis=0)/np.sqrt(Nc)

    t1 = ax1.plot(cutoffs, patient_cp, '-o', color='red', linewidth=2, label=subject)
    t2 = ax1.errorbar(cutoffs, healthy_cp, yerr=sem_cp, fmt='-o', color='blue', alpha=.8, linewidth=1, label='Healthy')
    t3 = ax1.bar(cutoffs-2, patient_bp, width=4, color=[1,0,0,0.4])
    t4 = ax1.bar(cutoffs+2, healthy_bp, width=4, color=[0,0,1,0.4])
    ax1.set_xlabel("$\omega_c$ (%)", fontsize=12), ax1.set_ylabel("$P_{\omega_c}$ (%)", fontsize=12)
    ax1.spines['right'].set_visible(False), ax1.spines['top'].set_visible(False)
    ax1.set_title(dict(info["tumor type & grade"])[subject]+' - '+dict(info["tumor location"])[subject], fontsize=12)
    ax1.set_xticks([0,20,40,60,80,100]), ax1.set_xticklabels(['0','20','40','60','80','100'])
    ax1.set_yticks([0,20,40,60,80,100]), ax1.set_yticklabels(['0','20','40','60','80','100'])
    ax1.set_xlim([0,105]), ax1.set_ylim([0,105])
    ax1.legend(frameon=False, fontsize=12, ncol=1, loc='upper left')

    # Total power
    _, p = ttest_1samp(healthy_nets_tp, patient_tp, alternative='two-sided')
    left, bottom, width, height = [0.35, 0.15, 0.125, 0.2]
    inset = fig.add_axes([left, bottom, width, height])
    inset.bar(0, patient_tp/healthy_tp, color=[1,0,0,0.4])
    inset.bar(1, 1, yerr=sem_tp/healthy_tp, color=[0,0,1,0.4], error_kw=dict(lw=1.5), ecolor='k', capsize=3, width=0.75, align='center')
    if p <=0.05:
        barplot_annotate_brackets(0, 1, p, [0,1],[1, patient_tp/healthy_tp], dh=0.25, barh=.01, fs=10)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_xticklabels([]), inset.set_ylabel("Relative $P_T$", fontsize=10) #inset.set_ylabel("$P_T=\sum_k |A_k|^2$", fontsize=10)
    inset.set_yticks([0,1]), inset.set_yticklabels(["0","1"])

    # ACF between DMNs
    """   
    time_significance_drop = np.zeros((Nc,))
    for i in range(Nc):
        dmn_pval = np.mean(DMN_regions_healthy_ACF_pval, axis=1)
        for j in range(dmn_pval.shape[-1]):
            if dmn_pval[i,j]>=0.048 and dmn_pval[i,j]<=0.052:
                time_significance_drop[i] = time_axis_healthy[i,j]
                print(dmn_pval[i,j],time_axis_healthy[i,j])
    """
    pat_acf = np.mean(DMN_ACF_patient, axis=0)
    healthy_acf, healthy_acf_sem = np.mean(np.mean(DMN_ACF_healthy, axis=1),axis=0), np.std(np.mean(DMN_ACF_healthy, axis=1),axis=0)/np.sqrt(Nc)
    left, bottom, width, height = [0.27, 0.5, 0.22, 0.25]
    inset = fig.add_axes([left, bottom, width, height])
    i_t1 = inset.plot(dmn_region_time_patient[0], pat_acf, color='red', linewidth=2, label=subject)
    inset.plot(time_axis_healthy[0], healthy_acf, color='blue', linewidth=1.7, alpha=0.8)
    inset.plot(time_axis_healthy[0], time_axis_healthy[0]*0, color='black', linewidth=1)
    inset.fill_between(time_axis_healthy[0], healthy_acf-healthy_acf_sem, healthy_acf+healthy_acf_sem,
        color = 'blue', alpha=0.3, edgecolor='blue', facecolor='blue'
    )
    inset.set_ylim([-.3,.3]), inset.set_yticks([-0.3,0,0.3]), inset.set_yticklabels(['-0.3', '0', '0.3'])
    inset.set_ylabel("ACF", fontsize=8), inset.set_xlabel("time (s)", fontsize=8)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.spines['bottom'].set_visible(False), inset.set_xlim([0, 450])
    inset.set_xticks(np.arange(0,450,100)), inset.tick_params(axis='both', which='both', length=0)

    # BOLD region i
    R = np.random.randint(0, 41)
    region = DMN_regions_patient[R]
    region_patient = DMN_bold_patient[R]
    region_control = DMN_bold_healthy[:,R,:]
    left, bottom, width, height = [0.52, 0.1, 0.235, 0.3]
    inset = fig.add_axes([left, bottom, width, height])
    i_t1 = inset.plot(dmn_region_time_patient[R], region_patient, color='red', linewidth=1, label=subject)
    for c in range(Nc):
        inset.plot(time_axis_healthy[c], region_control[c,:], color='blue', linewidth=0.35, alpha=0.3)
        inset.plot(time_axis_healthy[c], time_axis_healthy[c]*0, color='black', linewidth=1)
    inset.set_ylim([-5.1,5.1]), inset.set_yticks([-5,-2.5,0,2.5,5]), inset.set_yticklabels(['-5','-2.5','0','2.5','5'])
    inset.set_title("Region "+str(int(region)), fontsize=14)
    inset.set_ylabel("BOLD signal", fontsize=10), inset.set_xlabel("time (s)", fontsize=10)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.spines['bottom'].set_visible(False), inset.set_xlim([0, 450])
    inset.set_xticks(np.arange(0,450,100)), inset.tick_params(axis='both', which='both', length=0)

    # BOLD region j
    R = np.random.randint(0, 41)
    region = DMN_regions_patient[R]
    region_patient = DMN_bold_patient[R]
    region_control = DMN_bold_healthy[:,R,:]
    left, bottom, width, height = [0.755, 0.1, 0.235, 0.3]
    inset = fig.add_axes([left, bottom, width, height])
    i_t1 = inset.plot(dmn_region_time_patient[R], region_patient, color='red', linewidth=1, label=subject)
    for c in range(Nc):
        inset.plot(time_axis_healthy[c], region_control[c,:], color='blue', linewidth=0.35, alpha=0.3)
        inset.plot(time_axis_healthy[c], time_axis_healthy[c]*0, color='black', linewidth=1)
    inset.set_ylim([-5.1,5.1]), inset.set_yticks([-5,-2.5,0,2.5,5]), inset.set_yticklabels([])
    inset.set_title("Region "+str(int(region)), fontsize=14)
    inset.set_xlabel("time (s)", fontsize=10)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.spines['bottom'].set_visible(False), inset.set_xlim([0, 450])
    inset.set_xticks(np.arange(0,450,100)), inset.tick_params(axis='both', which='both', length=0)
    
    # DMN correlation nets
    left, bottom, width, height = [0.46, 0.54, 0.4, 0.4]
    inset = fig.add_axes([left, bottom, width, height])
    aa = inset.imshow(DMN_CorrNet_patient+np.identity(41), vmin=-1, vmax=1)
    inset.set_title("Subject", fontsize=14)
    inset.set_xticklabels([]), inset.set_yticklabels([])
    inset.tick_params(axis='both', which='both', length=0)
    cbar = fig.colorbar(aa, ax=inset, location='left', pad=0.02)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['-1', '1'])
    cbar.set_label("Correlation", fontsize=10, labelpad=0)

    net_healthy = np.mean(DMN_CorrNet_healthy, axis=0)
    left, bottom, width, height = [0.58, 0.54, 0.4, 0.4]
    inset = fig.add_axes([left, bottom, width, height])
    aa = inset.imshow(net_healthy+np.identity(41), vmin=-1, vmax=1)
    inset.set_title("Healthy", fontsize=14)
    inset.set_xticklabels([]), inset.set_yticklabels([])
    inset.tick_params(axis='both', which='both', length=0)

    # Complexity
    from matplotlib.ticker import FormatStrFormatter
    _, p = ttest_1samp(DMN_Complexity_healthy, DMN_Complexity_patient, alternative='two-sided')
    left, bottom, width, height = [0.89, 0.54, 0.1, 0.4]
    inset = fig.add_axes([left, bottom, width, height])
    inset.plot(0.01, DMN_Complexity_patient, 'ro')
    inset.errorbar(0.02, np.mean(DMN_Complexity_healthy), yerr=np.std(DMN_Complexity_healthy)/np.sqrt(Nc), fmt='o', color='blue', alpha=.8)
    inset.set_title("$\Theta$ Richness", fontsize=14)
    if p <=0.05:
        barplot_annotate_brackets(0, 1, p, [0.01,0.02],[np.mean(DMN_Complexity_healthy), DMN_Complexity_patient], dh=0.25, barh=.01, fs=10)
    inset.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.spines['bottom'].set_visible(False), inset.set_xlim([0, 0.03])   
    left_lim, right_lim =  np.min([np.mean(DMN_Complexity_healthy), DMN_Complexity_patient])-.08, np.max([np.mean(DMN_Complexity_healthy), DMN_Complexity_patient])+.08
    inset.set_ylim([left_lim, right_lim])
    inset.set_xticklabels([])
    inset.tick_params(axis='both', which='both', length=0)

    plt.savefig(subject_figures_path+subject+f'_{session}_DMN-comparisson.'+fig_fmt, dpi=1000)
    plt.close()

def group_analysis(
    dynamic_change, pcc_sim, overlap,
    distance, richness_change, dynamic_oedema_change_or,
    Nc, Np, session,
    power_change, healthy_power, 
    tumor_type, tumor_size, tumor_ventr, tumor_loc, tumor_grade, fig_fmt="png", tissue='Whole Tumor'
    ):
    # Wee keep the original direction of DAS
    dynamic_change_or = np.copy(dynamic_change) 
    mean_dynamic_change_or, sem_dynamic_change_or = np.mean(dynamic_change_or, axis=1), np.std(dynamic_change_or, axis=1)/np.sqrt(Nc)
    # Wee keep the original direction of complexity change
    richness_change_or = np.copy(richness_change)
    mean_richness_sim_or, sem_richness_sim_or = np.mean(richness_change_or, axis=1), np.std(richness_change_or, axis=1)/np.sqrt(Nc)
    richness_change = np.abs(richness_change)

    from scipy.stats import pearsonr, linregress
    print("================ FIg. 2 =================")
    fig, ax = plt.subplots(3,2,figsize=(13,15))
    plt.subplots_adjust(left=0.05,
                    bottom=0.04, 
                    right=0.96, 
                    top=0.98, 
                    wspace=0.1, 
                    hspace=0.15)
    ax = ax.flatten()
    ax[-1].remove(), ax[-2].remove()
    plt.gcf().text(0.009, 0.97, "A", fontsize=20, fontweight="bold")
    plt.gcf().text(0.009, 0.65, "B", fontsize=20, fontweight="bold")
    plt.gcf().text(0.009, 0.325, "C", fontsize=20, fontweight="bold")
    plt.gcf().text(0.33, 0.325, "D", fontsize=20, fontweight="bold")
    plt.gcf().text(0.64, 0.325, "E", fontsize=20, fontweight="bold")

    # A
    dynamic_change = np.abs(dynamic_change)
    mean_dynamic_change, sem_dynamic_change = np.mean(dynamic_change, axis=1), np.std(dynamic_change, axis=1)/np.sqrt(Nc)
    mean_pcc_sim, sem_pcc_sim = np.mean(pcc_sim, axis=1), np.std(pcc_sim, axis=1)/np.sqrt(Nc)
    r_dyn_pcc, p_dyn_pcc = pearsonr(mean_dynamic_change, mean_pcc_sim)
    slope, intercept, _, _, _ = linregress(mean_dynamic_change, mean_pcc_sim)
    fit = slope * np.linspace(8,40,400) + intercept
    ax[0].scatter(mean_dynamic_change, mean_pcc_sim)
    ax[0].errorbar(mean_dynamic_change, mean_pcc_sim, xerr=sem_dynamic_change_or, yerr=sem_pcc_sim, fmt='o')#, color="black", ecolor="black")
    ax[0].plot(np.linspace(8,40,400), fit, linewidth=1.5, color="tab:blue")
    ax[0].text(7,0.05,f"r = {round(r_dyn_pcc,3)} \np = {round(p_dyn_pcc,3)}", fontweight="bold", color="tab:blue")
    ax[0].spines['right'].set_visible(False), ax[0].spines['top'].set_visible(False)
    ax[0].set_yticks([0,.1,.2,.3,.4,.5]),ax[0].set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5'])
    ax[0].set_xlim([5,45]), ax[0].set_ylim([0,0.5])
    ax[0].set_ylabel('Node similarity (PCC)'), ax[0].set_xlabel("|DAS| (DMN)")
    positive_DAS, positive_pcc = mean_dynamic_change_or[mean_dynamic_change_or>=0], mean_pcc_sim[mean_dynamic_change_or>=0]
    negative_DAS, negative_pcc = mean_dynamic_change_or[mean_dynamic_change_or<=0], mean_pcc_sim[mean_dynamic_change_or<=0]
    r_dyn_pcc_pos, p_dyn_pcc_pos = permutation_test_correlation(positive_DAS, positive_pcc, num_permutations=5000)#pearsonr(positive_DAS, positive_pcc)
    r_dyn_pcc_neg, p_dyn_pcc_neg = permutation_test_correlation(negative_DAS, negative_pcc, num_permutations=5000)#pearsonr(negative_DAS, negative_pcc)
    slope_pos, intercept_pos, _, _, _ = linregress(positive_DAS, positive_pcc)
    fit_pos = slope_pos * np.linspace(-5,40,400) + intercept_pos
    slope_neg, intercept_neg, _, _, _ = linregress(negative_DAS, negative_pcc)
    fit_neg = slope_neg * np.linspace(-35,5,400) + intercept_neg
    left, bottom, width, height = [0.325, 0.88, 0.15, .10]
    inset = fig.add_axes([left, bottom, width, height])
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_xticks([0]), inset.set_xticklabels(["0"]), inset.set_yticklabels([])
    inset.xaxis.set_ticks_position('none'), inset.yaxis.set_ticks_position('none')
    inset.set_xlabel("DAS (DMN)"), inset.set_ylabel("PCC")
    inset.scatter(positive_DAS, positive_pcc, color='green', label="DAS>0", alpha=0.5)
    inset.scatter(negative_DAS, negative_pcc, color='orange', label="DAS<0", alpha=0.5)
    inset.plot(np.linspace(-5,40,400), fit_pos, color='green', linewidth=2)
    inset.plot(np.linspace(-40,5,400), fit_neg, color='orange', linewidth=2)
    inset.vlines(0,0,.4,colors="gray",linewidth=.5)
    inset.text(10,0.08,f"p = {round(p_dyn_pcc_pos/2,3)}", color="green", fontweight="bold")
    inset.text(10,0.03,f"p = {round(p_dyn_pcc_neg/2,3)}", color="orange", fontweight="bold")
    print(f"Correlation |DAS| with Node similarity:  r={r_dyn_pcc}, p={p_dyn_pcc} __ two-tailed exact test")
    print(f"Correlation DAS>0 with Node similarity:  r={r_dyn_pcc_pos}, p={p_dyn_pcc_pos/2} __ one-tailed permutation test 5000 resamples")
    print(f"Correlation DAS<0 with Node similarity:  r={r_dyn_pcc_neg}, p={p_dyn_pcc_neg/2} __ one-tailed permutation test 5000 resamples")
    print("---------------")


    #mean_dynamic_change, sem_dynamic_change = np.mean(dynamic_change, axis=1), np.std(dynamic_change, axis=1)/np.sqrt(Nc)
    mean_richness_sim, sem_richness_sim = np.mean(richness_change, axis=1), np.std(richness_change, axis=1)/np.sqrt(Nc)
    r_dyn_rich, p_dyn_rich = pearsonr(mean_dynamic_change_or, mean_richness_sim_or)
    # p_dyn_rich = p_dyn_rich / 2 # One sided
    slope, intercept, _, _, _ = linregress(mean_dynamic_change_or, mean_richness_sim_or)
    fit = slope * np.linspace(-30,40,400) + intercept
    ax[1].scatter(mean_dynamic_change_or, mean_richness_sim_or, color='b')
    ax[1].errorbar(mean_dynamic_change_or, mean_richness_sim_or, xerr=sem_dynamic_change_or, yerr=sem_richness_sim_or, fmt='o', color='b')
    ax[1].plot(np.linspace(-30,40,400), fit, color='b', linewidth=2)
    ax[1].text(-30,0.075,f"r = {round(r_dyn_rich,3)} \np = {round(p_dyn_rich,3)}", color='blue', fontweight="bold")
    ax[1].spines['right'].set_visible(False), ax[1].spines['top'].set_visible(False)
    ax[1].set_yticks([-.5,-.4,-.3,-.2,-.1,0,.1]),ax[1].set_yticklabels(['-0.5','-0.4','-0.3','-0.2','-0.1','0','0.1'])
    ax[1].set_xlim([-35,48]), ax[1].set_ylim([-0.55,0.15])
    ax[1].set_ylabel('$\Delta\Theta$ Complexity', color='b', fontweight='bold')
    ax[1].yaxis.label.set_color('blue'), ax[1].set_xlabel("DAS (DMN)")
    r_dyn_rich_abs, p_dyn_rich_abs = pearsonr(mean_dynamic_change, mean_richness_sim)
    # p_dyn_rich_abs = p_dyn_rich_abs / 2 # One sided
    slope, intercept, _, _, _ = linregress(mean_dynamic_change, mean_richness_sim)
    fit = slope * np.linspace(5,40,400) + intercept
    ax_bis = ax[1].twinx()
    ax_bis.scatter(mean_dynamic_change, mean_richness_sim, color='purple')
    ax_bis.errorbar(mean_dynamic_change, mean_richness_sim, xerr=sem_dynamic_change, yerr=sem_richness_sim, fmt='o', color='purple')
    ax_bis.plot(np.linspace(5,40,400), fit, color='purple', linewidth=2)
    ax_bis.text(32.5,0.3,f"r = {round(r_dyn_rich_abs,3)} \np = {round(p_dyn_rich_abs,3)}", color='purple', fontweight="bold")
    ax_bis.set_ylim([0,1]), ax_bis.set_yticks([0,.25,.5,.75,1]), ax_bis.set_yticklabels(['0','0.25','0.5','0.75','1'])
    ax_bis.set_ylabel('|$\Delta\Theta$| Complexity', color='purple', labelpad=-5, fontweight='bold')
    ax_bis.spines['top'].set_visible(False) #,ax_bis.spines['right'].set_visible(False)
    ax_bis.yaxis.label.set_color('purple')#, ax_bis.tick_params(direction="in")
    print(f"Correlation DAS with Richness: r={r_dyn_rich}, p={p_dyn_rich} __ two-tailed exact test")
    print(f"Correlation DAS with Richness (in absolutes): r={r_dyn_rich_abs}, p={p_dyn_rich_abs} __ two-tailed exact test")
    print("---------------")

    # B
    r_dyn_distance_abs, p_dyn_distance_abs = pearsonr(distance, mean_dynamic_change)
    r_dyn_distance, p_dyn_distance = pearsonr(distance, mean_dynamic_change_or)
    r_dyn_overlap_abs, p_dyn_overlap_abs = pearsonr(overlap, mean_dynamic_change)
    r_dyn_overlap, p_dyn_overlap = pearsonr(overlap, mean_dynamic_change_or)
    slope, intercept, _, _, _ = linregress(overlap, mean_dynamic_change_or)
    fit = slope * np.linspace(0,0.8,400) + intercept
    ax[2].scatter(mean_dynamic_change, distance, linewidths=2.5, color="cornflowerblue", edgecolors="royalblue")
    ax[2].spines['right'].set_visible(False), ax[2].spines['top'].set_visible(False)
    ax[2].set_ylabel('Distance (MNI units)'), ax[2].set_xlabel("|DAS| (DMN)")
    ax[2].text(20,92,f"r = {round(r_dyn_distance_abs,3)} \np = {round(p_dyn_distance_abs,3)}", color="royalblue")
    ax_bis = ax[2].twinx()    
    ax_bis.spines['right'].set_visible(False), ax_bis.spines['top'].set_visible(False)
    ax_bis.spines['left'].set_position(('data', 45)), ax_bis.yaxis.set_label_position('left'), ax_bis.yaxis.set_ticks_position('left')
    ax_bis.scatter(mean_dynamic_change+40, overlap, linewidths=2.5, color="lightsalmon", edgecolors="salmon")
    ax_bis.text(60,0.7125,f"r = {round(r_dyn_overlap_abs,3)} \np = {round(p_dyn_overlap_abs,3)}", color="salmon")
    ax_bis.set_xticks(np.arange(10,90,10)), ax_bis.set_xticklabels([10,20,30,40,10,20,30,40])
    ax_bis.set_ylim([-0.05,0.8]), ax_bis.set_yticks([0,0.2,0.4,0.6,0.8]), ax_bis.set_yticklabels([0,0.2,0.4,0.6,0.8])
    ax_bis.set_ylabel('Overlap (a. u.)')
    print(f"Correlation DAS with Spatial overlap: r={r_dyn_overlap}, p={p_dyn_overlap} __ two-tailed")
    print(f"Correlation |DAS| with Spatial overlap: r={r_dyn_overlap_abs}, p={p_dyn_overlap_abs} __ two-tailed")
    print(f"Correlation DAS with Euclidean distance: r={r_dyn_distance}, p={p_dyn_distance} __ two-tailed")
    print(f"Correlation |DAS| with Euclidean distance: r={r_dyn_distance_abs}, p={p_dyn_distance_abs} __ two-tailed")
    print("---------------")
    r_distance_pcc, p_distance_pcc = pearsonr(distance, mean_pcc_sim)
    r_overlap_pcc, p_overlap_pcc = pearsonr(overlap, mean_pcc_sim)
    print(f"Correlation PCC with Spatial overlap: r={r_overlap_pcc}, p={p_overlap_pcc} __ two-tailed")
    print(f"Correlation PCC with Distance: r={r_distance_pcc}, p={p_distance_pcc} __ two-tailed")
    print("---------------")

    #mean_dynamic_change, sem_dynamic_change = np.mean(dynamic_change_or, axis=1), np.std(dynamic_change_or, axis=1)/np.sqrt(Nc)
    mean_dynamic_odema_sim, sem_dynamic_odema_sim = np.mean(dynamic_oedema_change_or, axis=1), np.std(dynamic_oedema_change_or, axis=1)/np.sqrt(Nc)
    r_dyn_dyn, p_dyn_dyn = pearsonr(mean_dynamic_change_or, mean_dynamic_odema_sim)
    r_dyn_dyn_abs, p_dyn_dyn_abs = pearsonr(mean_dynamic_change, np.mean(np.abs(dynamic_oedema_change_or), axis=1))
    result = linregress(mean_dynamic_change_or, mean_dynamic_odema_sim)
    slope, intercept, slope_err, intercept_err = result.slope, result.intercept, result.stderr, result.intercept_stderr
    ci=1.96 # Confidence interval
    slope_max, slope_min, intercept_max, intercept_min = (slope+ci*slope_err), (slope-ci*slope_err), (intercept+ci*intercept_err), (intercept-ci*intercept_err)
    x1, x2 = np.linspace(0,40,400), np.linspace(-35,0,400) 
    fit = slope * np.linspace(-35,40,400) + intercept
    fit_max_max, fit_max_min, fit_min_max, fit_min_min = slope_max * x1 + intercept_max, slope_max * x2 + intercept_min, slope_min * x2 + intercept_max, slope_min * x1 + intercept_min
    ax[3].scatter(mean_dynamic_change_or, mean_dynamic_odema_sim, color="royalblue")
    ax[3].errorbar(mean_dynamic_change_or, mean_dynamic_odema_sim, xerr=sem_dynamic_change, yerr=sem_dynamic_odema_sim, fmt='o', color="black", ecolor="black")
    ax[3].plot(np.linspace(-35,40,400), fit, color='salmon', linewidth=2)
    ax[3].fill_between(x2, fit_min_max, fit_max_min, color="lightsalmon", alpha=0.1, linewidth=0)
    ax[3].fill_between(x1, fit_max_max, fit_min_min, color="lightsalmon", alpha=0.1, linewidth=0)
    plot_pval = " < 0.001" if p_dyn_dyn<0.001 else f" = {round(r_dyn_dyn,3)}"
    ax[3].text(-35,40,f"r = {round(r_dyn_dyn,3)} \np{plot_pval}", fontweight="bold", color="salmon")
    ax[3].spines['right'].set_visible(False), ax[3].spines['top'].set_visible(False)
    ax[3].set_xlim([-40,45]), ax[3].set_ylim([-40,55])
    ax[3].set_ylabel(f'DAS ({tissue})'), ax[3].set_xlabel("DAS (DMN)")
    print(f"Correlation DAS (DMN vs Tumor): r={r_dyn_dyn}, p={p_dyn_dyn} __ two-tailed")
    print(f"Correlation |DAS| (DMN vs Tumor): r={r_dyn_dyn_abs}, p={p_dyn_dyn_abs} __ two-tailed")
    print("---------------")

    # C
    mean_power_H = np.mean(healthy_power, axis=1)
    pc, dasc = np.mean(power_change, axis=1)/mean_power_H, np.mean(dynamic_oedema_change_or, axis=1)
    pc_abs, dasc_abs = np.abs(pc), np.abs(dasc)
    _, p_pc = ttest_1samp(pc, 0, alternative='two-sided')
    _, p_pc_abs = ttest_1samp(pc_abs, 0, alternative='greater')
    _, p_pc_abs_U = mannwhitneyu(pc_abs, 0, alternative='greater')
    _, p_dasc = ttest_1samp(dasc, 0, alternative='two-sided')
    _, p_dasc_abs = ttest_1samp(dasc_abs, 0, alternative='greater')
    _, p_dasc_abs_U = mannwhitneyu(dasc_abs, 0, alternative='greater')
    print(f"Power change: p={p_pc} __ two-tailed")
    print(f"Abs Power change: pT={p_pc_abs}, pU={p_pc_abs_U} __ one-tailed")
    print(f"DAS change: p={p_dasc} __ two-tailed")
    print(f"ABS DAS change: pT={p_dasc_abs}, pU={p_dasc_abs_U} __ one-tailed")
    print("---------------")

    # Plotting Relative Power
    left, bottom, width, height = [0.05, 0.025, 0.125, .3]
    inset = fig.add_axes([left, bottom, width, height])
    bx_data = np.array([pc, pc_abs]).T
    positions = [2, 2.75]
    for i in range(2):   
        xdata = positions[i]+0*bx_data[:,i]+np.random.normal(0,0.1,size=bx_data[:,i].shape)
        inset.scatter(xdata, bx_data[:,i], s=10)
    bx = inset.boxplot(bx_data, 
        positions=positions, widths=0.5, patch_artist=True,
        showmeans=False, showfliers=False,
        medianprops={'color':'black', 'linewidth':1.8},
        #meanprops={'marker':'s', 'markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':6}
    )
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False), inset.spines['bottom'].set_visible(False)  
    inset.set_xticks([2, 2.75]), inset.set_xticklabels(["$\Delta$P", "|$\Delta$P|"], fontsize=10), inset.xaxis.set_ticks_position('none') 
    for b in bx['boxes']:
        b.set_edgecolor('k') # or try 'black'
        b.set_facecolor([0.3,0.3,0.6,0.2])
        b.set_linewidth(1.5)
    plt.gcf().text(0.136, 0.035, "*", fontsize=13)

    # Plotting DAS
    left, bottom, width, height = [0.2, 0.025, 0.14, .3]
    inset = fig.add_axes([left, bottom, width, height])
    bx_data = np.array([dasc, dasc_abs]).T
    positions = [2, 2.75]
    for i in range(2):   
        xdata = positions[i]+0*bx_data[:,i]+np.random.normal(0,0.1,size=bx_data[:,i].shape)
        inset.scatter(xdata, bx_data[:,i], s=10)
    bx = inset.boxplot(bx_data, 
        positions=positions, widths=0.5, patch_artist=True,
        showmeans=False, showfliers=False,
        medianprops={'color':'black', 'linewidth':1.8},
        #meanprops={'marker':'s', 'markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':6}
    )
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False), inset.spines['bottom'].set_visible(False)  
    inset.set_xticks([2, 2.75]), inset.set_xticklabels(["DAS", "|DAS|"], fontsize=10), inset.xaxis.set_ticks_position('none') 
    for b in bx['boxes']:
        b.set_edgecolor('k') # or try 'black'
        b.set_facecolor([0.3,0.3,0.6,0.2])
        b.set_linewidth(1.5)
    plt.gcf().text(0.2965, 0.135, "*", fontsize=13)
    plt.gcf().text(0.2965, 0.125, "*", fontsize=13)
    plt.gcf().text(0.2965, 0.115, "*", fontsize=13)
    
    # D 
    from sklearn.decomposition import FastICA
    from sklearn.cluster import KMeans
    embed = FastICA(n_components=2).fit_transform(dynamic_oedema_change_or)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embed)
    print(f"K-means score: {kmeans.score(embed)}")
    left, bottom, width, height = [0.38, 0.025, 0.26, .3]
    inset = fig.add_axes([left, bottom, width, height])
    inset.scatter(embed[:,0][kmeans.labels_==0],embed[:,1][kmeans.labels_==0], color='blue')
    inset.scatter(embed[:,0][kmeans.labels_==1],embed[:,1][kmeans.labels_==1], color='red')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axvline(x=0, color='black', linewidth=1)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.spines['left'].set_visible(False), inset.spines['bottom'].set_visible(False)
    inset.xaxis.set_ticks_position('none'), inset.yaxis.set_ticks_position('none')

    # E
    # Correlations DAS with Features
    r_das_type, p_das_type = pearsonr(dasc, tumor_type)
    r_das_size, p_das_size = pearsonr(dasc, tumor_size)
    r_das_ventr, p_das_ventr = pearsonr(dasc, tumor_ventr)
    r_das_loc, p_das_loc = pearsonr(dasc, tumor_loc)
    r_das_grade, p_das_grade = pearsonr(dasc, tumor_grade)
    print(f"DAS with type: r={r_das_type} and p-val={p_das_type}")
    print(f"DAS with size: r={r_das_size} and p-val={p_das_size}")
    print(f"DAS with ventricular: r={r_das_ventr} and p-val={p_das_ventr}")
    print(f"DAS with location: r={r_das_loc} and p-val={p_das_loc}")
    print(f"DAS with grade: r={r_das_grade} and p-val={p_das_grade}")
    print("---------------")
    # Groups
    menin = np.array([i for i,j in zip(dasc, tumor_type) if j==1])
    gliom = np.array([i for i,j in zip(dasc, tumor_type) if j==2])
    small = np.array([i for i,j in zip(dasc, tumor_size) if j<np.percentile(tumor_size,50)])
    large = np.array([i for i,j in zip(dasc, tumor_size) if j>np.percentile(tumor_size,50)]) 
    perivent = np.array([i for i,j in zip(dasc, tumor_ventr) if j==2])
    non_peri = np.array([i for i,j in zip(dasc, tumor_ventr) if j==1])
    frontal = np.array([i for i,j in zip(dasc, tumor_loc) if j==1])
    other = np.array([i for i,j in zip(dasc, tumor_loc) if j==2])
    grade_I = np.array([i for i,j in zip(dasc, tumor_grade) if j==1])
    grade_II = np.array([i for i,j in zip(dasc, tumor_grade) if j==2])
    # Tests
    _, pT_type = ttest_ind(menin, gliom, equal_var=False, alternative='two-sided')
    _, pU_type = mannwhitneyu(menin, gliom, alternative='two-sided')
    _, pT_size = ttest_ind(small, large, equal_var=False, alternative='two-sided')
    _, pU_size = mannwhitneyu(small, large, alternative='two-sided')
    _, pT_ventr = ttest_ind(perivent, non_peri, equal_var=False, alternative='greater')
    _, pU_ventr = mannwhitneyu(perivent, non_peri, alternative='greater')
    _, pT_loc = ttest_ind(frontal, other, equal_var=False, alternative='two-sided')
    _, pU_loc = mannwhitneyu(frontal, other, alternative='two-sided')
    _, pT_grade = ttest_ind(grade_I, grade_II, equal_var=False, alternative='two-sided')
    _, pU_grade = mannwhitneyu(grade_I, grade_II, alternative='two-sided')
    print(f"Type (2s): pT={pT_type} and pU={pU_type}")
    print(f"Size (2s): pT={pT_size} and pU={pU_size}")
    print(f"Ventricular (1s): pT={pT_ventr} and pU={pU_ventr}")
    print(f"Location (2s): pT={pT_loc} and pU={pU_loc}")
    print(f"Grade (2s): pT={pT_grade} and pU={pU_grade}")
    print("---------------")
    # Plots
    left, bottom, width, height = [0.69, 0.15, 0.28, .165]
    inset = fig.add_axes([left, bottom, width, height])
    inset.plot([0,1], [np.mean(menin), np.mean(gliom)], '--s', label='Meningioma', markersize=5)
    inset.plot([0,1], [np.mean(small), np.mean(large)], '--s', label='Small', markersize=5)
    inset.plot([0,1], [np.mean(perivent), np.mean(non_peri)], '-*', label='Periventricular', markersize=8)
    inset.plot([0,1], [np.mean(frontal), np.mean(other)], '--s', label='Frontal', markersize=5)
    inset.plot([0,1], [np.mean(grade_I), np.mean(grade_II)], '--s', label='Grade I', markersize=5)
    inset.set_ylim([-.5,13])
    inset.legend(frameon=False,ncols=2, loc='upper right'), inset.set_xlim([-0.3,1.3]), inset.set_xticks([0,1]), inset.set_xticklabels(["YES", "NO"])
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_ylabel("DAS",labelpad=0)

    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    x = np.linspace(-40,40,200)
    log_reg_grade = LogisticRegression(random_state=0).fit(dasc[:,np.newaxis], tumor_ventr-1)
    log_reg_curve = expit(x[:,np.newaxis]*log_reg_grade.coef_+log_reg_grade.intercept_).ravel() + 1
    Det_coef = log_reg_grade.score(dasc[:,np.newaxis], tumor_ventr-1)
    left, bottom, width, height = [0.69, 0.025, 0.28, .10]
    inset = fig.add_axes([left, bottom, width, height])
    inset.scatter(dasc,tumor_ventr,color='k')
    inset.plot(x, log_reg_curve, linewidth=0.8, color='r', label=f"R$^2$ = {Det_coef}")
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_xlabel("DAS",labelpad=0), inset.set_yticks([1.25,2.07]), inset.set_yticklabels(["Non-PV", "PV"], rotation = 90)
    inset.set_ylim([0.5,2.5]), inset.yaxis.set_ticks_position('none')
    inset.legend(frameon=False)

    # Regressions
    """ from sklearn.linear_model import LinearRegression, LogisticRegression
    from scipy.stats import zscore
    from .methods import coef_pvals
    tumor_size = zscore(tumor_size)
    predictors = np.vstack((tumor_type, tumor_size))
    #predictors = np.vstack((predictors, tumor_ventr))
    predictors = np.vstack((predictors, tumor_loc))
    predictors = np.vstack((predictors, tumor_grade)).T
    reg = LinearRegression()
    reg.fit(predictors, zscore(dasc))
    coef_linear, score_linear = coef_pvals(predictors, zscore(dasc), reg)
    print(coef_linear)
    print("Score multilinear: ", score_linear)
    #x = np.linspace(-2.5,2.5,50)
    log_reg_type = LogisticRegression(random_state=0).fit(zscore(dasc)[:,np.newaxis], tumor_type-1)
    print("Score Logistic type: ", log_reg_type.score(zscore(dasc)[:,np.newaxis], tumor_type-1))
    log_reg_grade = LogisticRegression(random_state=0).fit(zscore(dasc)[:,np.newaxis], tumor_grade-1)
    print("Score Logistic grade: ", log_reg_grade.score(zscore(dasc)[:,np.newaxis], tumor_grade-1))

    plt.gcf().text(0.43, 0.31,
    "$\Delta$DAS = $a_1 \cdot$ H + $a_2 \cdot$ S + $a_3 \cdot$ V + $a_4 \cdot$ L + $a_5 \cdot$ G ",
    fontsize=17)
    left, bottom, width, height = [0.39, 0.033, 0.28, .262]
    inset = fig.add_axes([left, bottom, width, height])
    inset.scatter(zscore(dasc),tumor_type)
    inset.spines['right'].set_visible(False), inset.spines['top'].set_visible(False)
    inset.set_xlabel("$\Delta$DAS (Oedema)"), inset.set_yticks([1.15,2.05]), inset.set_yticklabels(["Meningioma", "Glioma"], rotation = 90)
    inset.set_ylim([0.8,2.2]), inset.yaxis.set_ticks_position('none')"""

    plt.savefig(f"../RESULTS/figures/functional/{session}/global-analysis_{session}."+fig_fmt, dpi=1000)
    plt.close()

