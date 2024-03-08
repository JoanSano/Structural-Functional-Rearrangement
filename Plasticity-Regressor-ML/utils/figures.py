import colorsys
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import probplot, pearsonr, permutation_test, ttest_ind, mannwhitneyu, f_oneway, kruskal, linregress

from models.methods import f_test, to_array

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

def boxplot(png_path, args, mse, mse_z, mae_z, pcc_z, cs_z, kl_z, js_z, PAT_subjects):
    subject_to_follow_max = np.argmax(mse) # Highest zscore of reconstruction error
    subject_to_follow_min = np.argmin(mse) # Lowest zscore of reconstruction error
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(left=0.08,
                    bottom=0.08, 
                    right=0.98, 
                    top=0.92)
    plt.gcf().text(0.01, 0.96, "A", fontsize=20, fontweight="bold")
    positions = [2, 4, 6, 8, 10, 12]
    bx_data = np.array([mse_z, mae_z, pcc_z, cs_z, kl_z, js_z]).T
    for i in range(6):   
        xdata = positions[i]+0*bx_data[:,i]+np.random.normal(0,0.15,size=bx_data[:,i].shape)
        ax.scatter(xdata, bx_data[:,i], s=15)
        if i==0:
            ax.plot(xdata[subject_to_follow_max], bx_data[subject_to_follow_max,i], 'k*', markersize=10, label=PAT_subjects[subject_to_follow_max])
            ax.plot(xdata[subject_to_follow_min], bx_data[subject_to_follow_min,i], 'ks', markersize=7, label=PAT_subjects[subject_to_follow_min])
        else:
            ax.plot(xdata[subject_to_follow_max], bx_data[subject_to_follow_max,i], 'k*', markersize=10)
            ax.plot(xdata[subject_to_follow_min], bx_data[subject_to_follow_min,i], 'ks', markersize=7)
    bx = ax.boxplot(bx_data, 
        positions=positions, widths=1.5, patch_artist=True,
        showmeans=False, showfliers=False,
        medianprops={'color':'black', 'linewidth':1.8},
        #meanprops={'marker':'s', 'markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':6}
    )
    ax.set_ylabel('zscore', fontsize=16), ax.set_xticklabels(['MSE', 'MAE', 'PCC', 'CS', 'KL', 'JS'], fontsize=16)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    for b in bx['boxes']:
        b.set_edgecolor('k') # or try 'black'
        b.set_facecolor([0.3,0.3,0.6,0.2])
        b.set_linewidth(1.5)
    plt.legend(loc=9, frameon=True, fontsize=10, ncol=2, bbox_to_anchor=(0.5,1.1))
    plt.savefig(png_path+args.model+'_boxplot.svg', dpi=1000)
    plt.savefig(png_path+args.model+'_boxplot.eps', dpi=1000)

def normality_plots(png_path, mse, mae, pcc, cs, kl, js, args, PAT_subjects):
    fig, ax = plt.subplots(figsize=(6,4.5))
    plt.subplots_adjust(left=0.09,
                    bottom=0.1, 
                    right=0.98, 
                    top=0.98)
    norm_mse, fit_mse = probplot(mse)
    norm_mae, fit_mae = probplot(mae)
    norm_pcc, fit_pcc = probplot(pcc)
    norm_cs, fit_cs = probplot(cs)
    norm_kl, fit_kl = probplot(kl)
    norm_js, fit_js = probplot(js)
    
    ax.scatter(norm_mse[0], norm_mse[1], s=10, label='MSE')
    ax.scatter(norm_mae[0], norm_mae[1], s=10, label='MAE')
    ax.scatter(norm_pcc[0], norm_pcc[1], s=10, label='PCC')
    ax.scatter(norm_cs[0], norm_cs[1], s=10, label='CS')
    ax.scatter(norm_kl[0], norm_kl[1], s=10, label='KL')
    ax.scatter(norm_js[0], norm_js[1], s=10, label='JS')

    ax.plot(np.arange(-2,2,0.01), np.arange(-2,2,0.01), 'k--', linewidth=1)
    
    ax.set_title(""), ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12), ax.set_ylabel('zscore', fontsize=12)
    ax.set_xticks([-2,-1,0,1,2]), ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
    plt.legend(loc=2, frameon=False, fontsize=10, ncol=2)
    plt.savefig(png_path+args.model+'_normality.svg', dpi=1000)
    plt.savefig(png_path+args.model+'_normality.eps', dpi=1000)

    print("Linear fits for the normality plots:")
    print("MSE: r = ",fit_mse[2])
    print("MAE: r = ",fit_mae[2])
    print("PCC: r = ",fit_pcc[2])
    print("CS: r = ",fit_cs[2])
    print("KL: r = ",fit_kl[2])
    print("JS: r = ",fit_js[2])
    print("=====================================")

def size_correlation(figs_path, args, mae, pcc, tumor_sizes, PAT_subjects, alpha=0.05):
    #################################
    ### Correlation size vs error ###
    #################################
    from models.methods import grubbs_test

    tm_size = to_array(tumor_sizes)
    # Dropping 3 biggest
    tm_3drop = tm_size[tm_size.argsort()[:-3][::-1]]
    mae_3drop = mae[tm_size.argsort()[:-3][::-1]]
    pcc_3drop = pcc[tm_size.argsort()[:-3][::-1]]
    _, p_grub_large,_ = grubbs_test(np.concatenate((tm_3drop, [np.mean(tm_size[tm_size.argsort()[-3:][::-1]])]), axis=0))
    print("Mean of the 3 largest tumor is outlier with p = {:.4}".format(p_grub_large))
    print("=============================")
    # Dropping 4 biggest
    tm_4drop = tm_size[tm_size.argsort()[:-4][::-1]]
    mae_4drop = mae[tm_size.argsort()[:-4][::-1]]
    pcc_4drop = pcc[tm_size.argsort()[:-4][::-1]]

    r_mae, p_mae = pearsonr(mae, tm_size)
    r_pcc, p_pcc = pearsonr(pcc, tm_size)
    r_mae_3drop, p_mae_3drop = pearsonr(mae_3drop, tm_3drop)
    r_pcc_3drop, p_pcc_3drop = pearsonr(pcc_3drop, tm_3drop)
    r_mae_4drop, p_mae_4drop = pearsonr(mae_4drop, tm_4drop)
    r_pcc_4drop, p_pcc_4drop = pearsonr(pcc_4drop, tm_4drop)
    p_mae, p_pcc, p_mae_3drop, p_pcc_3drop, p_mae_4drop, p_pcc_4drop = p_mae/2, p_pcc/2, p_mae_3drop/2, p_pcc_3drop/2, p_mae_4drop/2, p_pcc_4drop/2

    # Permutation tests 
    samples = 500
    statistic = lambda x, y: pearsonr(x,y)[0]
    permu_pcc = permutation_test((pcc, tm_size), statistic, n_resamples=samples, alternative='less')
    permu_3pcc = permutation_test((pcc_3drop, tm_3drop), statistic, n_resamples=samples, alternative='less')
    permu_4pcc = permutation_test((pcc_4drop, tm_4drop), statistic, n_resamples=samples, alternative='less')
    permu_mae = permutation_test((mae, tm_size), statistic, n_resamples=samples, alternative='greater')
    permu_3mae = permutation_test((mae_3drop, tm_3drop), statistic, n_resamples=samples, alternative='greater')
    permu_4mae = permutation_test((mae_4drop, tm_4drop), statistic, n_resamples=samples, alternative='greater')
    
    fig, ax = plt.subplots(figsize=(5,3.5))
    plt.subplots_adjust(left=0.12,
                    bottom=0.12, 
                    right=0.98, 
                    top=0.98)
    plt.gcf().text(0.018, 0.95, "B", fontsize=15, fontweight="bold")
    plt.scatter(tm_size, pcc, s=10, label='r = ' + str(round(r_pcc,3)))
    plt.scatter(tm_3drop, pcc_3drop, s=10, label='r = ' + str(round(r_pcc_3drop,3)))
    slope, intercept, _, _, _ = linregress(tm_size, pcc)
    fit = slope * np.linspace(0,90,400) + intercept
    plt.plot(np.linspace(0,90,400), fit, color='b', linewidth=0.5)
    slope, intercept, _, _, _ = linregress(tm_3drop, pcc_3drop)
    fit = slope * np.linspace(0,55,400) + intercept
    plt.plot(np.linspace(0,55,400), fit, color='orange', linewidth=0.5)

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_ylim([0.76,0.88]), ax.set_yticks([0.76,0.78,0.80,0.82,0.84,0.86,0.88]), ax.set_yticklabels([0.76,0.78,0.80,0.82,0.84,0.86,0.88], fontsize=8)
    ax.set_xticks([0,20,40,60,80]), ax.set_xticklabels(['0','20','40','60','80'], fontsize=8)
    ax.set_xlabel('Tumor size (cm$^3$)', fontsize=8), ax.set_ylabel('PCC', fontsize=8)
    plt.legend(loc=4, frameon=True, fontsize=8)
    plt.savefig(figs_path+args.model+'_size-effects.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_size-effects.eps', dpi=1000)

    print("Correlations with tumor size:")
    print("PCC: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f}".format(r_pcc, p_pcc, permu_pcc.pvalue))
    print("PCC: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (3 dropped)".format(r_pcc_3drop, p_pcc_3drop, permu_3pcc.pvalue))
    print("PCC: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (4 dropped)".format(r_pcc_4drop, p_pcc_4drop, permu_4pcc.pvalue))
    print("MAE: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f}".format(r_mae, p_mae, permu_mae.pvalue))
    print("MAE: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (3 dropped)".format(r_mae_3drop, p_mae_3drop, permu_3mae.pvalue))
    print("MAE: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (4 dropped)".format(r_mae_4drop, p_mae_4drop, permu_4mae.pvalue))
    print("=============================")

    #################################
    ### Tumor size between groups ###
    #################################
    # Dividing metrics percentiles of the tumor size 
    small = np.array([j for j,i in zip(pcc,tm_size) if i<np.percentile(tm_size,50)])
    large = np.array([j for j,i in zip(pcc,tm_size) if i>np.percentile(tm_size,50)])
    mean_small, std_small = np.mean(small), np.std(small)/np.sqrt(small.shape[0])
    mean_large, std_large = np.mean(large), np.std(large)/np.sqrt(large.shape[0])
    _, p_var = f_test(small, large)
    eq_var = True if p_var>alpha else False
    _, pT = ttest_ind(small, large, equal_var=eq_var, alternative='greater')
    _, pU = mannwhitneyu(small, large, alternative='greater')
    print("One-sided differences in PCC between 2 tumor size groups (splitted by P50):")
    print("Small group: mean = {:.4f} +/- std = {:.4f}".format(mean_small, std_small))
    print("Large group: mean = {:.4f} +/- std = {:.4f}".format(mean_large, std_large))
    print("T-test p = {:.4f}".format(pT))
    print("U-test p = {:.4f}".format(pU))
    print("=============================")

    fig, ax = plt.subplots(figsize=(5,4))
    plt.subplots_adjust(bottom=0.08, 
                    right=0.98, 
                    top=0.98)
    ax.bar([1,2], [mean_small, mean_large], 
        yerr=[std_small, std_large],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )
    barplot_annotate_brackets(0, 1, '*', [1,2], [mean_small+0.06, mean_large+0.06], dh=0.01, barh=.001, fs=10)
    xdata = 1+np.random.normal(0,0.08,size=small.shape)
    ax.scatter(xdata, small, s=10, color="black")
    xdata = 2+np.random.normal(0,0.08,size=large.shape)
    ax.scatter(xdata, large, s=10, color="black", marker="s")

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.68,0.90]), ax.set_yticks([0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90])
    ax.set_xticklabels(['Size<'+str(np.percentile(tm_size,50))+'cm$^3$','Size>'+str(np.percentile(tm_size,50))+'cm$^3$'])
    ax.set_xticks([1,2]), ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-size.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_tumor-size.eps', dpi=1000)

    groups = [
        np.array([j for j,i in zip(pcc,tm_size) if i<=np.percentile(tm_size,33)]),
        np.array([j for j,i in zip(pcc,tm_size) if i>np.percentile(tm_size,33) and i<=np.percentile(tm_size,67)]),
        np.array([j for j,i in zip(pcc,tm_size) if i>np.percentile(tm_size,67)])
    ]
    _, pA = f_oneway(*groups)
    _, pKW = kruskal(*groups)
    print("Differences in PCC between 3 tumor size groups (splitted by P33-66):")
    print("Small group: mean = {:.4f} +/- std = {:.4f}".format(np.mean(groups[0]), np.std(groups[0])/np.sqrt(groups[0].shape[0])))
    print("Medium group: mean = {:.4f} +/- std = {:.4f}".format(np.mean(groups[1]), np.std(groups[1])/np.sqrt(groups[1].shape[0])))
    print("Large group: mean = {:.4f} +/- std = {:.4f}".format(np.mean(groups[2]), np.std(groups[2]/np.sqrt(groups[2].shape[0]))))
    print("ANOVA p = {:.4f}".format(pA))
    print("KRUSKAL-WALLIS p = {:.4f}".format(pKW))
    print("=============================")

def type_effects(figs_path, args, mae, pcc, tumor_types, PAT_subjects, alpha=0.05):
    meningioma, glioma = [] , []
    for s in range(len(PAT_subjects)):
        if 'gioma' in tumor_types[PAT_subjects[s]]:
            meningioma.append([pcc[s], mae[s]])
        else:
            glioma.append([pcc[s], mae[s]])
    meningioma = np.array(meningioma, dtype=np.float64)
    glioma = np.array(glioma, dtype=np.float64)

    mean_pcc_menin, mean_mae_menin = np.mean(meningioma[:,0]), np.mean(meningioma[:,1])
    mean_pcc_gliom, mean_mae_gliom = np.mean(glioma[:,0]), np.mean(glioma[:,1])
    std_pcc_menin, std_mae_menin = np.std(meningioma[:,0])/len(meningioma), np.std(meningioma[:,1])/len(meningioma)
    std_pcc_gliom, std_mae_gliom = np.std(glioma[:,0])/len(glioma), np.std(glioma[:,1])/len(glioma)
    
    # PCC
    _, p_var = f_test(meningioma[:,0], glioma[:,0])
    eq_var = True if p_var>alpha else False
    _, p_pcc_T = ttest_ind(meningioma[:,0], glioma[:,0], equal_var=eq_var, alternative='two-sided')
    _, p_pcc_U = mannwhitneyu(meningioma[:,0], glioma[:,0], alternative='two-sided')
    # MAE
    _, p_var = f_test(meningioma[:,1], glioma[:,1])
    eq_var = True if p_var>alpha else False
    _, p_mae_T = ttest_ind(meningioma[:,1], glioma[:,1], equal_var=eq_var, alternative='two-sided')
    _, p_mae_U = mannwhitneyu(meningioma[:,1], glioma[:,1], alternative='two-sided')

    print("Error with tumor type: (MEAN +- SEM)")
    print("Meningioma: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_menin, std_pcc_menin, mean_mae_menin, std_mae_menin))
    print("Glioma: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_gliom, std_pcc_gliom, mean_mae_gliom, std_mae_gliom))
    print("Differences between tumor types, T-test:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_T, p_pcc_T/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_T, p_mae_T/2))
    print("Differences between tumor types, Mann-Whitney:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_U, p_pcc_U/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_U, p_mae_U/2))
    print("=============================")

    fig, ax = plt.subplots(figsize=(3,4))
    plt.subplots_adjust(left=0.21,
                    bottom=0.08, 
                    right=0.98, 
                    top=0.98)
    plt.gcf().text(0.007, 0.96, "C", fontsize=15, fontweight="bold")
    ax.bar([1,2], [mean_pcc_menin, mean_pcc_gliom], 
        yerr=[std_pcc_menin, std_pcc_gliom],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )
    xdata = 1+np.random.normal(0,0.08,size=meningioma[:,0].shape)
    ax.scatter(xdata, meningioma[:,0], s=10, color="black")
    xdata = 2+np.random.normal(0,0.08,size=glioma[:,0].shape)
    ax.scatter(xdata, glioma[:,0], s=10, color="black", marker="s")

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.64,0.88]), ax.set_yticks([0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86, 0.88])
    ax.set_xticks([1,2]), ax.set_xticklabels(['Meningioma', 'Glioma']), ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-type.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_tumor-type.eps', dpi=1000)

def location_effects(figs_path, args, mae, pcc, tumor_locs, PAT_subjects, alpha=0.05):
    frontal, non_periven = [] , []
    for s in range(len(PAT_subjects)):
        if 'frontal' in tumor_locs[PAT_subjects[s]].lower():
            frontal.append([pcc[s], mae[s]])
        else:
            non_periven.append([pcc[s], mae[s]])
    frontal = np.array(frontal, dtype=np.float64)
    non_periven = np.array(non_periven, dtype=np.float64)

    mean_pcc_front, mean_mae_front = np.mean(frontal[:,0]), np.mean(frontal[:,1])
    mean_pcc_oth, mean_mae_oth = np.mean(non_periven[:,0]), np.mean(non_periven[:,1])
    std_pcc_front, std_mae_front = np.std(frontal[:,0])/len(frontal), np.std(frontal[:,1])/len(frontal)
    std_pcc_oth, std_mae_oth = np.std(non_periven[:,0])/len(non_periven), np.std(non_periven[:,1])/len(non_periven)
    
    # PCC
    _, p_var = f_test(frontal[:,0], non_periven[:,0])
    eq_var = True if p_var>alpha else False
    _, p_pcc_T = ttest_ind(frontal[:,0], non_periven[:,0], equal_var=eq_var, alternative='two-sided')
    _, p_pcc_U = mannwhitneyu(frontal[:,0], non_periven[:,0], alternative='two-sided')
    # MAE
    _, p_var = f_test(frontal[:,1], non_periven[:,1])
    eq_var = True if p_var>alpha else False
    _, p_mae_T = ttest_ind(frontal[:,1], non_periven[:,1], equal_var=eq_var, alternative='two-sided')
    _, p_mae_U = mannwhitneyu(frontal[:,1], non_periven[:,1], alternative='two-sided')

    print("Error with tumor location: (MEAN +- SEM)")
    print("Frontal: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_front, std_pcc_front, mean_mae_front, std_mae_front))
    print("Other: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_oth, std_pcc_oth, mean_mae_oth, std_mae_oth))
    print("Differences between tumor locations, T-test:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_T, p_pcc_T/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_T, p_mae_T/2))
    print("Differences between tumor locations, Mann-Whitney:")
    print("PCC one-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_U, p_pcc_U/2))
    print("MAE one-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_U, p_mae_U/2))
    print("=============================")

    fig, ax = plt.subplots(figsize=(3,4))
    plt.gcf().text(0.007, 0.96, "D", fontsize=15, fontweight="bold")
    plt.subplots_adjust(left=0.21,
                    bottom=0.08, 
                    right=0.98, 
                    top=0.98)
    ax.bar([1,2], [mean_pcc_front, mean_pcc_oth], 
        yerr=[std_pcc_front, std_pcc_oth],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )
    xdata = 1+np.random.normal(0,0.08,size=frontal[:,0].shape)
    ax.scatter(xdata,frontal[:,0], s=10, color="black")
    xdata = 2+np.random.normal(0,0.08,size=non_periven[:,0].shape)
    ax.scatter(xdata, non_periven[:,0], s=10, color="black", marker="s")

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.66,0.88]), ax.set_yticks([0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88])#, ax.set_yticklabels(['0.76','0.78','0.80','0.82'])
    ax.set_xticks([1,2]), ax.set_xticklabels(['Frontal', 'Other']), ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-loc.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_tumor-loc.eps', dpi=1000)

def periventricularity_effects(figs_path, args, mae, pcc, tumor_ventricular, PAT_subjects, alpha=0.05):
    perivent, other = [] , []
    for s in range(len(PAT_subjects)):
        if 'yes' in tumor_ventricular[PAT_subjects[s]].lower():
            perivent.append([pcc[s], mae[s]])
        else:
            other.append([pcc[s], mae[s]])
    perivent = np.array(perivent, dtype=np.float64)
    other = np.array(other, dtype=np.float64)

    mean_pcc_front, mean_mae_front = np.mean(perivent[:,0]), np.mean(perivent[:,1])
    mean_pcc_oth, mean_mae_oth = np.mean(other[:,0]), np.mean(other[:,1])
    std_pcc_front, std_mae_front = np.std(perivent[:,0])/len(perivent), np.std(perivent[:,1])/len(perivent)
    std_pcc_oth, std_mae_oth = np.std(other[:,0])/len(other), np.std(other[:,1])/len(other)
    
    # PCC
    _, p_var = f_test(perivent[:,0], other[:,0])
    eq_var = True if p_var>alpha else False
    _, p_pcc_T = ttest_ind(perivent[:,0], other[:,0], equal_var=eq_var, alternative='two-sided')
    _, p_pcc_U = mannwhitneyu(perivent[:,0], other[:,0], alternative='two-sided')
    # MAE
    _, p_var = f_test(perivent[:,1], other[:,1])
    eq_var = True if p_var>alpha else False
    _, p_mae_T = ttest_ind(perivent[:,1], other[:,1], equal_var=eq_var, alternative='two-sided')
    _, p_mae_U = mannwhitneyu(perivent[:,1], other[:,1], alternative='two-sided')

    print("Error with tumor periventricularity: (MEAN +- SEM)")
    print("Perivent: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_front, std_pcc_front, mean_mae_front, std_mae_front))
    print("Non-PV: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_oth, std_pcc_oth, mean_mae_oth, std_mae_oth))
    print("Differences between tumor periventricularities, T-test:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_T, p_pcc_T/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_T, p_mae_T/2))
    print("Differences between tumor periventricularities, Mann-Whitney:")
    print("PCC one-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_U, p_pcc_U/2))
    print("MAE one-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_U, p_mae_U/2))
    print("=============================")

    fig, ax = plt.subplots(figsize=(5,4))
    plt.subplots_adjust(bottom=0.08, 
                    right=0.98, 
                    top=0.98)
    ax.bar([1,2], [mean_pcc_front, mean_pcc_oth], 
        yerr=[std_pcc_front, std_pcc_oth],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.78,0.84]), ax.set_yticks([0.78,0.80,0.82,0.84]), ax.set_yticklabels(['0.78','0.80','0.82','0.84'])
    ax.set_xticks([1,2]), ax.set_xticklabels(['PV', 'Non-PV']), ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-PV.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_tumor-PV.eps', dpi=1000)

def grade_effects(figs_path, args, mae, pcc, tumor_grade, PAT_subjects, alpha=0.05):
    grade2_3, other = [] , []
    for s in range(len(PAT_subjects)):
        if 'II' in tumor_grade[PAT_subjects[s]]:#.lower():
            grade2_3.append([pcc[s], mae[s]])
        else:
            other.append([pcc[s], mae[s]])
    grade2_3 = np.array(grade2_3, dtype=np.float64)
    other = np.array(other, dtype=np.float64)

    mean_pcc_front, mean_mae_front = np.mean(grade2_3[:,0]), np.mean(grade2_3[:,1])
    mean_pcc_oth, mean_mae_oth = np.mean(other[:,0]), np.mean(other[:,1])
    std_pcc_front, std_mae_front = np.std(grade2_3[:,0])/len(grade2_3), np.std(grade2_3[:,1])/len(grade2_3)
    std_pcc_oth, std_mae_oth = np.std(other[:,0])/len(other), np.std(other[:,1])/len(other)
    
    # PCC
    _, p_var = f_test(grade2_3[:,0], other[:,0])
    eq_var = True if p_var>alpha else False
    _, p_pcc_T = ttest_ind(grade2_3[:,0], other[:,0], equal_var=eq_var, alternative='two-sided')
    _, p_pcc_U = mannwhitneyu(grade2_3[:,0], other[:,0], alternative='two-sided')
    # MAE
    _, p_var = f_test(grade2_3[:,1], other[:,1])
    eq_var = True if p_var>alpha else False
    _, p_mae_T = ttest_ind(grade2_3[:,1], other[:,1], equal_var=eq_var, alternative='two-sided')
    _, p_mae_U = mannwhitneyu(grade2_3[:,1], other[:,1], alternative='two-sided')

    print("Error with tumor grade: (MEAN +- SEM)")
    print(mean_pcc_front.shape, mean_pcc_oth.shape)
    print("Grade II-III: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_front, std_pcc_front, mean_mae_front, std_mae_front))
    print("Grade I: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_oth, std_pcc_oth, mean_mae_oth, std_mae_oth))
    print("Differences between tumor grade, T-test:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_T, p_pcc_T/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_T, p_mae_T/2))
    print("Differences between tumor grade, Mann-Whitney:")
    print("PCC one-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_U, p_pcc_U/2))
    print("MAE one-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_U, p_mae_U/2))
    print("=============================")

    fig, ax = plt.subplots(figsize=(2.5,4))
    plt.subplots_adjust(left=0.05,
                    bottom=0.08, 
                    right=0.98, 
                    top=0.98)
    ax.bar([1,2], [mean_pcc_oth, mean_pcc_front], 
        yerr=[std_pcc_oth, std_pcc_front],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )
    xdata = 1+np.random.normal(0,0.08,size=other[:,0].shape)
    ax.scatter(xdata,other[:,0], s=10, color="black")
    xdata = 2+np.random.normal(0,0.08,size=grade2_3[:,0].shape)
    ax.scatter(xdata, grade2_3[:,0], s=10, color="black", marker="s")
    #barplot_annotate_brackets(0, 1, '*', [1,2], [mean_pcc_oth, mean_pcc_front], dh=0.005, barh=.001, fs=10)

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.64,0.88]), ax.set_yticks([0.74,0.76,0.78,0.80,0.82,0.84,0.86, 0.88]), ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_xticks([1,2]), ax.set_xticklabels(['Grade I', 'Grade II-III'])#, ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-grade.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_tumor-grade.eps', dpi=1000)

def plot_degree_distribution(figs_path, args, degree_file):
    import pandas as pd
    from scipy.ndimage.filters import uniform_filter1d
    degree_list = pd.read_csv(degree_file, sep='\t')

    dgs = np.linspace(0,1000,1001)
    dgs = np.array(dgs, dtype=np.int64)
    distributions = np.zeros((len(degree_list["Subject"]), dgs.shape[0]))

    # Predictions
    fig, ax = plt.subplots(figsize=(8,8))
    for i, (sub, dg_dist) in enumerate(zip(degree_list["Subject"], degree_list["Predicted"])):
        dg_dist_new = []
        k = 0
        for s in dg_dist.split(' '):
            dg_dist_new.append(s.strip('\n').strip('.').strip('[').strip(']'))
        for s in range(len(dg_dist_new)):
            try:
                distributions[i, k] = float(dg_dist_new[s])
                k += 1
            except:
                pass   
        ax.plot(dgs,distributions[i], color=[np.random.random(),np.random.random(),np.random.random()], linewidth=1, alpha=0.1)

    to_plot = uniform_filter1d(np.mean(distributions, axis=0), 5)
    ax.plot(dgs,to_plot, color='k', linewidth=2.5, label='Predicted')

    # Ground truth
    for i, (sub, dg_dist) in enumerate(zip(degree_list["Subject"], degree_list["Ground Truth"])):
        dg_dist_new = []
        k = 0
        for s in dg_dist.split(' '):
            dg_dist_new.append(s.strip('\n').strip('.').strip('[').strip(']'))
        for s in range(len(dg_dist_new)):
            try:
                distributions[i, k] = float(dg_dist_new[s])
                k += 1
            except:
                pass
        #ax.plot(dgs,distributions[i], color=[np.random.random(),np.random.random(),np.random.random()], linewidth=1, alpha=0.1, label=sub)

    to_plot = uniform_filter1d(np.mean(distributions, axis=0), 5)
    ax.plot(dgs,to_plot, color='r', linestyle='--', linewidth=2.5, label='Ground Truth')

    ax.set_xlabel('log(1+$\omega$)', fontsize=20), ax.set_ylabel('Probability', fontsize=20)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xlim([0, 500])#, #ax.set_ylim([0, 0.035])
    ax.set_title("FCNET", fontsize=20)
    plt.legend(loc='upper right', frameon=False, fontsize=20)

    plt.savefig(figs_path+args.model+'_degree-probs.svg', dpi=1000)
    plt.savefig(figs_path+args.model+'_degree-probs.eps', dpi=1000)

def prior_stats(thetas, entropies, mods, folder, prior_1, prior_2):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    plt.subplots_adjust(left=0.09,
                    bottom=0.14, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.6, 
                    hspace=1)

    ax[0].plot(thetas, entropies, 'k', linewidth=2)
    ax[0].set_ylabel('Entropy', fontsize=10), ax[0].set_xlabel("Threshold", fontsize=10)
    ax[0].vlines(0.2, 8.7, 9.3, colors='r', linestyles='dotted')
    ax[0].vlines(0.7, 8.7, 9.3, colors='green', linestyles='dotted')
    ax[0].set_xticks([0,0.2,0.7,1]), ax[0].set_yticks([8.8,8.9,9.0,9.1,9.2])
    ax[0].set_xlim([0,1]), ax[0].set_ylim([8.75,9.21])
    ax_bis = ax[0].twinx()
    ax_bis.plot(thetas, mods, 'b', linewidth=2)
    ax_bis.set_ylabel('Modularity', fontsize=10, color='blue')

    y1, binEdges = np.histogram(prior_1, bins=10)
    y2, _ = np.histogram(prior_2, bins=10)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax[1].plot(bincenters, y1, color='red')
    ax[1].plot(bincenters, y2, color='green')
    ax[1].set_xticks([0,0.5,1]), ax[1].set_xlabel('P($\lambda$=1)')
    ax[1].set_ylabel('Counts')
    plt.savefig(folder + '/prior_stats.svg', dpi=1000)