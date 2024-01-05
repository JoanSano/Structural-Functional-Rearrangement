import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, pearsonr
from scipy.spatial.distance import jensenshannon
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

from utils.data import prepare_structural_methods, prepare_structural_healthy, match_structural_methods
from utils.paths import check_path

def pcc(x, y, dim=-1):
    """
    Inputs:
        output: network output tensor of size (N, Features) N>1! 
        target: tensor of size (N, Features)
    Outputs:
        cc: correlation coefficient of each feature - tensor of size (Features,)
        mean_cc: mean correlation coefficient - scalar 
    """
    vx = x - torch.mean(x, dim=dim)
    vy = y - torch.mean(y, dim=dim)
    cc = torch.sum(vx * vy, dim=dim) / (torch.sqrt(torch.sum(vx ** 2, dim=dim)) * torch.sqrt(torch.sum(vy ** 2, dim=dim)))
    mean_cc = torch.mean(cc)
    std_cc = torch.std(cc)
    return cc, mean_cc, std_cc

def degree_distribution(G, rois, maximum_degree=1000, d_dg=1.):
    """
    Returns the probability distribution and the degrees in the graph. 
    Inputs:
        flattened: flattened graph
        rois: number of nodes
        maximum_degree: (int) maximum degree to which spans the probability
        d_dg: degree interval upon which the probability refers to (float)
    Outputs:
        prob: probability distribution of each degree in the network (numpy array)
        dgs: degrees present in the network until maximum_degree
    """
    degree_prob = np.zeros((int(maximum_degree//d_dg),))
    dgs = np.arange(0, maximum_degree+1)
    D_G = [jj for _,jj in G.degree(weight='weight')]
    #probs = (np.bincount([jj for _,jj in D_G])/rois)
    #dgs = np.unique([jj for _,jj in D_G])
    for d in range(maximum_degree):
        d_inf, d_sup = dgs[d], dgs[d+1]
        degree_prob[d] = np.sum((D_G>d_inf)*(D_G<d_sup))
    return degree_prob/rois, dgs

def KL_JS_divergences(G1, G2, rois, eps=1e-8):
    """ Computes the KL and JS Divergences between two degree distributions.
    Input:
        input: degree distribution of the input graph
        target: degree distribution of the target graph
        rois: number of nodes in the graph (to be used in the degree computation)
        eps: float to avoid log(0)
    Output:
        KL: divergence (torch scalar) 
        JS: divergence (torch scalar)
    """

    input_degree, _ = degree_distribution(G1, rois)
    target_degree, _ = degree_distribution(G2, rois)
    kl = np.sum(target_degree*np.log(target_degree+eps) - target_degree*np.log(input_degree+eps))
    js = jensenshannon(input_degree, target_degree)
    return kl, js

if __name__ == '__main__':
    # Load data
    CONTROL, C_subjects = prepare_structural_healthy(
        '../Data/structural/graphs/hybrid/', rois=170, sessions=['preop','postop'], norm=True, flatten=False, del_rois=[35,36,81,82], dtype=torch.float64
    )
    Hybrid, H_subjects = prepare_structural_methods(
        '../Data/structural/graphs/hybrid/', rois=170, session='preop', norm=True, flatten=False, del_rois=[35,36,81,82], dtype=torch.float64
    )
    MSMT, MS_subjects = prepare_structural_methods(
        '../Data/structural/graphs/hybrid/', rois=170, session='preop', norm=True, flatten=False, del_rois=[35,36,81,82], dtype=torch.float64
    )
    Hybrid, MSMT, Patients = match_structural_methods(Hybrid, H_subjects, MSMT, MS_subjects)
    NC, NH, NMS = CONTROL.shape[0], Hybrid.shape[0], MSMT.shape[0]

    metrics = ['Cosine-Similarity', 'Pearson-Correlation', 'Mean-Squared-Error']#, 'MSE-Clustering-Coeff', 'JS', 'MSE-Global-Eff', 'MSE-N_components']
    N_metrics = len(metrics)

    figures_path = check_path('RESULTS/figures/structural/')
    numerical_path = check_path('RESULTS/numerical/structural/')
    
    ######################
    ### Tumor features ###
    ######################
    # Loading patient information and REORDERING in the same order!!!
    N_features = 5
    info = pd.read_csv('../Data/participants.tsv', sep='\t')
    info = info[info["participant_id"].str.contains("CON") == False]
    info.set_index(info.participant_id, inplace=True)
    info.drop(['participant_id'], axis=1, inplace=True)
    info.index.name = None
    tumor_sizes = np.array([dict(info["tumor size (cub cm)"])[k] for k in Patients])
    tumor_types = np.array([1 if 'ningioma' in dict(info["tumor type & grade"])[k] else 2 for k in Patients])
    tumor_locs = np.array([1 if 'Frontal' in dict(info["tumor location"])[k] else 2 for k in Patients])
    tumor_grade = np.array([2 if 'II' in dict(info["tumor type & grade"])[k] else 1 for k in Patients])
    tumor_ventricles = np.array([2 if 'yes' in dict(info["ventricles"])[k] else 1 for k in Patients])
    """ tumor_sizes_post = np.array([dict(info["tumor size (cub cm)"])[k] for k in Patients_post])
    tumor_types_post = np.array([1 if 'ningioma' in dict(info["tumor type & grade"])[k] else 2 for k in Patients_post])
    tumor_locs_post = np.array([1 if 'Frontal' in dict(info["tumor location"])[k] else 2 for k in Patients_post])
    tumor_grade_post = np.array([2 if 'II' in dict(info["tumor type & grade"])[k] else 1 for k in Patients_post])
    tumor_ventricles_post = np.array([2 if 'yes' in dict(info["ventricles"])[k] else 1 for k in Patients_post]) """

    """ pre = torch.cat((pre1,pre2), dim=0)
    Patients_post = Patients.copy()
    Patients.extend(PAT2)
    Nc, Np, Npost = CONTROL.shape[0], pre.shape[0], post.shape[0] """

    ###############
    ### Metrics ###
    ###############
    if not os.path.exists(numerical_path+'delta_pcc-networks.npy'):
        CS, PCC, MSE = torch.zeros((NC,NC)), torch.zeros((NC,NC)), torch.zeros((NC,NC))
        Healthy = torch.zeros((NC*(NC-1)//2,N_metrics))
        NetAlterations_H = torch.zeros((NH,NC,N_metrics))
        NetAlterations_MS = torch.zeros((NMS,NC,N_metrics))
        k = 0
        for i in range(NC):

            for pat in range(NH):
                ### hybrid pipeline ###
                # Numerical Similarity
                cosine = F.cosine_similarity(CONTROL[i], Hybrid[pat], dim=-1).mean()
                pearson = pcc(CONTROL[i], Hybrid[pat], dim=-1)[1].mean()
                distance = F.mse_loss(CONTROL[i], Hybrid[pat], reduction='mean').mean()
                NetAlterations_H[pat,i,:] = torch.tensor([cosine, pearson, distance])

            for pat in range(NMS):
                ### msmt pipeline ###
                # Numerical Similarity
                cosine = F.cosine_similarity(CONTROL[i], MSMT[pat], dim=-1).mean()
                pearson = pcc(CONTROL[i], MSMT[pat], dim=-1)[1].mean()
                distance = F.mse_loss(CONTROL[i], MSMT[pat], reduction='mean').mean()
                NetAlterations_MS[pat,i,:] = torch.tensor([cosine, pearson, distance])

            ### Healthy cross-subject ###
            CS[i,i], PCC[i,i], MSE[i,i] = 1, 1, 1
            for j in range(i+1,NC):
                # Numerical Similarity
                CS[i,j] = F.cosine_similarity(CONTROL[i], CONTROL[j], dim=-1).mean()
                PCC[i,j] = pcc(CONTROL[i], CONTROL[j],dim=-1)[1]
                MSE[i,j] = F.mse_loss(CONTROL[i], CONTROL[j], reduction='mean')
                CS[j,i], PCC[j,i], MSE[j,i] = CS[i,j], PCC[i,j], MSE[i,j]

        delta_pcc = NetAlterations_H[...,1]-NetAlterations_MS[...,1]
        np.save(numerical_path+'delta_pcc-networks.npy', delta_pcc)
    else:
        delta_pcc = np.load(numerical_path+'delta_pcc-networks.npy')

    mean_delta_pcc = delta_pcc.mean()
    _, pvals = wilcoxon(delta_pcc, axis=1)
    _, pval_mean = wilcoxon(mean_delta_pcc, alternative='greater')
    print(f"Mean delta PCC: {mean_delta_pcc} greater than zero with p-val: {pval_mean}")

    fig, ax = plt.subplots(1,1,figsize=(6,8))
    ax.remove()

    left, bottom, width, height = [0.04, 0.025, 0.9, 0.925]
    ax1 = fig.add_axes([left, bottom, width, height])
    aa = ax1.imshow(delta_pcc)
    ax1.set_title("$\Delta$PCC", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks(np.arange(25)), ax1.set_yticklabels(Patients)
    for i, ytick in enumerate(ax1.get_yticklabels()):
        if pvals[i] < 0.01:
            color = 'red'
        elif pvals[i] > 0.01 and pvals[i] < 0.05:
            color = 'blue'
        else:
            color = 'black'
        ytick.set_color(color)
    cbar = fig.colorbar(aa, ax=ax1, location='right', pad=0.03)
    plt.savefig(figures_path+"delta_PCC-methods.svg", dpi=1000)
    plt.close()

    # ANOVA and tests
    print("deltaPCC and size: ", pearsonr(delta_pcc.mean(axis=1),tumor_sizes))
    data = pd.DataFrame(
        {'deltaPCC': delta_pcc.mean(axis=1), 'size': tumor_sizes, 'type': tumor_types, 'grade': tumor_grade}
    )
    model = ols(
        """ deltaPCC ~ size + C(type) + C(grade) +
            size:C(type) + C(type):C(grade) + size:C(grade) +
            size:C(type):C(grade) """, data=data
    ).fit()
    ANOVA_3way = sm.stats.anova_lm(model, typ=2)
    print(ANOVA_3way)

    ################################
    ### Compute network measures ###
    ################################
    """ measures = 'datasets/structural_network-measures_method-hybrid_'
    if os.path.exists(measures):
        # Loading measures and REORDERING in the same order as prepare_data() does!!!
        # Healthy
        net_measures = pd.read_csv(measures+'healthy.tsv', sep='\t')
        net_measures.set_index(net_measures.Subject, inplace=True)
        net_measures.drop(['Subject'], axis=1, inplace=True)
        cc_healthy = np.array([dict(net_measures["Clustering Coefficient"])[k] for k in C_subjects])
        ge_healthy = np.array([dict(net_measures["Global Efficiency"])[k] for k in C_subjects])
        comp_healthy = np.array([dict(net_measures["Num Components"])[k] for k in C_subjects])

        # Preop
        net_measures_pre = pd.read_csv(measures+'ses-preop.tsv', sep='\t')
        net_measures_pre.set_index(net_measures_pre.Subject, inplace=True)
        net_measures_pre.drop(['Subject'], axis=1, inplace=True)
        cc_pre = np.array([dict(net_measures_pre["Clustering Coefficient"])[k] for k in Patients])
        ge_pre = np.array([dict(net_measures_pre["Global Efficiency"])[k] for k in Patients])
        comp_pre = np.array([dict(net_measures_pre["Num Components"])[k] for k in Patients])

        # Postop
        net_measures_post = pd.read_csv(measures+'ses-postop.tsv', sep='\t')
        net_measures_post.set_index(net_measures_post.Subject, inplace=True)
        net_measures_post.drop(['Subject'], axis=1, inplace=True)
        cc_post = np.array([dict(net_measures_pre["Clustering Coefficient"])[Patients[k]] for k in range(Npost)])
        ge_post = np.array([dict(net_measures_pre["Global Efficiency"])[Patients[k]] for k in range(Npost)])
        comp_post = np.array([dict(net_measures_pre["Num Components"])[Patients[k]] for k in range(Npost)])
    else:
        # Healthy
        net_measures = pd.DataFrame(columns=['Subject', 'Clustering Coefficient', 'Global Efficiency', 'Num Components'])
        for i in range(Nc):
            G_healthy = nx.from_numpy_array(np.array(CONTROL[i]))
            healthy_clustering = nx.average_clustering(G_healthy, weight='weight')
            healthy_GE = nx.global_efficiency(G_healthy)
            components = nx.number_connected_components(G_healthy)
            net_measures.loc[len(net_measures.index)] = [C_subjects[i], healthy_clustering, healthy_GE, components]
        net_measures.to_csv(measures+'healthy.tsv', sep='\t', index=False)

        # Patients
        net_measures_pre = pd.DataFrame(columns=['Subject', 'Clustering Coefficient', 'Global Efficiency', 'Num Components'])
        net_measures_post = pd.DataFrame(columns=['Subject', 'Clustering Coefficient', 'Global Efficiency', 'Num Components'])
        for pat in range(Np):
            G_patient_pre = nx.from_numpy_array(np.array(pre[pat]))
            cluster = nx.average_clustering(G_patient_pre, weight='weight')
            global_eff = nx.global_efficiency(G_patient_pre)
            components = nx.number_connected_components(G_patient_pre)
            net_measures_pre.loc[len(net_measures_pre.index)] = [Patients[pat], cluster, global_eff, components]
            if pat < Npost:
                G_patient_post = nx.from_numpy_array(np.array(post[pat]))
                cluster = nx.average_clustering(G_patient_post, weight='weight')
                global_eff = nx.global_efficiency(G_patient_post)
                components = nx.number_connected_components(G_patient_post)
                net_measures_post.loc[len(net_measures_post.index)] = [Patients[pat], cluster, global_eff, components]

        net_measures_pre.to_csv(measures+'ses-preop.tsv', sep='\t', index=False)
        net_measures_post.to_csv(measures+'ses-postop.tsv', sep='\t', index=False)

        print("Writing network measures to /datasets. Run again the code.")
        quit()

    ###############
    ### Metrics ###
    ###############
    CS, PCC, MSE = torch.zeros((Nc,Nc)), torch.zeros((Nc,Nc)), torch.zeros((Nc,Nc))
    Healthy = torch.zeros((Nc*(Nc-1)//2,N_metrics))
    Damage_pre = torch.zeros((Np,Nc,N_metrics))
    Damage_post = torch.zeros((Npost,Nc,N_metrics))
    k = 0
    for i in range(Nc):
        ### Healthy Reference ###
        G_healthy = nx.from_numpy_array(np.array(CONTROL[i]))
        healthy_clustering = cc_healthy[i]
        healthy_GE = ge_healthy[i]
        N_components = comp_healthy[i]

        for pat in range(Np):
            ### ses-preop ###
            # Numerical Similarity
            cosine = F.cosine_similarity(CONTROL[i], pre[pat], dim=-1).mean()
            pearson = pcc(CONTROL[i], pre[pat], dim=-1)[1].mean()
            distance = F.mse_loss(CONTROL[i], pre[pat], reduction='mean').mean()
            # Structural Similarity
            G_patient_pre = nx.from_numpy_array(np.array(pre[pat]))
            clust_diff = (healthy_clustering - cc_pre[pat])**2
            _, JS_weight = KL_JS_divergences(G_healthy, G_patient_pre, rois=166)
            global_eff_diff = (healthy_GE - ge_pre[pat])**2
            components_diff = (N_components - comp_pre[pat])**2
            Damage_pre[pat,i,:] = torch.tensor([cosine, pearson, distance, clust_diff, JS_weight, global_eff_diff, components_diff])
            
            if pat < Npost:
                ### ses-postop ###
                # Numerical Similarity
                cosine = F.cosine_similarity(CONTROL[i], post[pat], dim=-1).mean()
                pearson = pcc(CONTROL[i], post[pat], dim=-1)[1].mean()
                distance = F.mse_loss(CONTROL[i], post[pat], reduction='mean').mean()
                # Structural Similarity
                G_patient_post = nx.from_numpy_array(np.array(post[pat]))
                clust_diff = (healthy_clustering - cc_post[pat])**2
                _, JS_weight = KL_JS_divergences(G_healthy, G_patient_post, rois=166)
                global_eff_diff = (healthy_GE - ge_post[pat])**2
                components_diff = (N_components - comp_post[pat])**2
                Damage_post[pat,i,:] = torch.tensor([cosine, pearson, distance, clust_diff, JS_weight, global_eff_diff, components_diff])

        ### Healthy cross-subject ###
        CS[i,i], PCC[i,i], MSE[i,i] = 1, 1, 1
        for j in range(i+1,Nc):
            # Numerical Similarity
            CS[i,j] = F.cosine_similarity(CONTROL[i], CONTROL[j], dim=-1).mean()
            PCC[i,j] = pcc(CONTROL[i], CONTROL[j],dim=-1)[1]
            MSE[i,j] = F.mse_loss(CONTROL[i], CONTROL[j], reduction='mean')
            CS[j,i], PCC[j,i], MSE[j,i] = CS[i,j], PCC[i,j], MSE[i,j]
            # Structural Similarity
            G_bis = nx.from_numpy_array(np.array(CONTROL[j]))
            clust_diff = (healthy_clustering - cc_healthy[j])**2
            global_eff_diff = (healthy_GE - ge_healthy[j])**2
            components_diff = (N_components - comp_healthy[j])**2
            Healthy[k] = torch.tensor([CS[i,j], PCC[i,j], MSE[i,j], clust_diff, KL_JS_divergences(G_healthy, G_bis, rois=166)[1], global_eff_diff, components_diff])
            k += 1

    ##################################
    ### Clustering Metric features ###
    ##################################
    n_components, n_clusters = 2, 2
    colors = np.random.rand(N_metrics, n_clusters, 3)
    ### Sesssion preop 
    Embeded_pre, kmeans_pre = np.zeros((N_metrics,Np,2)), []
    groups_pre = []
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(21,16))
    ax = ax.flat
    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.99, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.1)
    for metric in range(N_metrics):
        # Dimensionality reduction & Clustering
        Embeded_pre[metric,:,:] = PCA(n_components=n_components).fit_transform(Damage_pre[:,:,metric])
        if metric==3 or metric==5:
            Embeded_pre[metric,:,:] = PCA(n_components=n_components).fit_transform(Damage_pre[:,:,metric])
        else:
            Embeded_pre[metric,:,:] = FastICA(n_components=n_components, whiten='unit-variance').fit_transform(Damage_pre[:,:,metric])
        kmeans_pre.append(KMeans(n_clusters=n_clusters, random_state=0).fit(Embeded_pre[metric,:,:]))
        # Groups  
        summary = pd.DataFrame(columns=['Features', 'Cluster 1', 'Cluster 2', 'SEM C1', 'SEM C2'])
        groups_pre.append([
            [tumor_sizes[kmeans_pre[metric].labels_==0], tumor_sizes[kmeans_pre[metric].labels_==1]],
            [tumor_types[kmeans_pre[metric].labels_==0], tumor_types[kmeans_pre[metric].labels_==1]],
            [tumor_locs[kmeans_pre[metric].labels_==0], tumor_locs[kmeans_pre[metric].labels_==1]],
            [tumor_grade[kmeans_pre[metric].labels_==0], tumor_grade[kmeans_pre[metric].labels_==1]],
            [tumor_ventricles[kmeans_pre[metric].labels_==0], tumor_ventricles[kmeans_pre[metric].labels_==1]]
            ])
        summary.loc[len(summary.index)] = ["Size", groups_pre[metric][0][0].mean(), groups_pre[metric][0][1].mean(), groups_pre[metric][0][0].std()/np.sqrt(len(groups_pre[metric][0][0])), groups_pre[metric][0][1].std()/np.sqrt(len(groups_pre[metric][0][1]))]
        summary.loc[len(summary.index)] = ["Histology", groups_pre[metric][1][0].mean(), groups_pre[metric][1][1].mean(), groups_pre[metric][1][0].std()/np.sqrt(len(groups_pre[metric][1][0])), groups_pre[metric][1][1].std()/np.sqrt(len(groups_pre[metric][1][1]))]
        summary.loc[len(summary.index)] = ["Location", groups_pre[metric][2][0].mean(), groups_pre[metric][2][1].mean(), groups_pre[metric][2][0].std()/np.sqrt(len(groups_pre[metric][2][0])), groups_pre[metric][2][1].std()/np.sqrt(len(groups_pre[metric][2][1]))]
        summary.loc[len(summary.index)] = ["Grade", groups_pre[metric][3][0].mean(), groups_pre[metric][3][1].mean(), groups_pre[metric][3][0].std()/np.sqrt(len(groups_pre[metric][3][0])), groups_pre[metric][3][1].std()/np.sqrt(len(groups_pre[metric][3][1]))]
        summary.loc[len(summary.index)] = ["Ventricles", groups_pre[metric][4][0].mean(), groups_pre[metric][4][1].mean(), groups_pre[metric][4][0].std()/np.sqrt(len(groups_pre[metric][4][0])), groups_pre[metric][3][1].std()/np.sqrt(len(groups_pre[metric][4][1]))]
        
        for cluster in range(n_clusters):            
            ax[metric].scatter(
                Embeded_pre[metric,:,:][kmeans_pre[metric].labels_==cluster,0], Embeded_pre[metric,:,:][kmeans_pre[metric].labels_==cluster,1],
                color=colors[metric,cluster,:] #np.random.rand(3,)
                )
        ax[metric].set_title(
            'Metric: {}'.format(metrics[metric]),
            fontsize=14)
        summary.to_csv(numerical_path+'metric-{}_ses-preop.tsv'.format(metrics[metric]), sep='\t', index=False)
    plt.savefig(figures_path+'clusters_ses-preop.png', dpi=100)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(21,16))
    ax = ax.flat
    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.99, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.1)
    for metric in range(N_metrics):
        for feature in range(N_features):
            mean_g1, mean_g2 = np.mean(Damage_pre[:,kmeans_pre[metric].labels_==0,metric]), np.mean(Damage_pre[:,kmeans_pre[metric].labels_==1,metric])
            sem_g1 = np.std(Damage_pre[:,kmeans_pre[metric].labels_==0,metric])/np.sqrt(len(Damage_pre[:,kmeans_pre[metric].labels_==0,metric]))
            sem_g2 = np.std(Damage_pre[:,kmeans_pre[metric].labels_==1,metric])/np.sqrt(len(Damage_pre[:,kmeans_pre[metric].labels_==1,metric]))
    plt.savefig(figures_path+'clustered-groups_ses-preop.png', dpi=100)

    ### Sesssion postop 
    Embeded_post, kmeans_post = np.zeros((N_metrics,Npost,2)), []
    groups_post = []
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(21,16))
    ax = ax.flat
    plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.99, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.1)
    for metric in range(N_metrics):
        # Dimensionality reduction & Clustering
        #Embeded_post[metric,:,:] = PCA(n_components=n_components).fit_transform(Damage_post[:,:,metric])
        if metric==3 or metric==5:
            Embeded_post[metric,:,:] = PCA(n_components=n_components).fit_transform(Damage_post[:,:,metric])
        else:
            Embeded_post[metric,:,:] = FastICA(n_components=n_components, whiten='unit-variance').fit_transform(Damage_post[:,:,metric])
        kmeans_post.append(KMeans(n_clusters=n_clusters, random_state=0).fit(Embeded_post[metric,:,:]))
        # Groups  
        summary = pd.DataFrame(columns=['Features', 'Cluster 1', 'Cluster 2', 'SEM C1', 'SEM C2'])
        groups_post.append([
            [tumor_sizes_post[kmeans_post[metric].labels_==0], tumor_sizes_post[kmeans_post[metric].labels_==1]],
            [tumor_types_post[kmeans_post[metric].labels_==0], tumor_types_post[kmeans_post[metric].labels_==1]],
            [tumor_locs_post[kmeans_post[metric].labels_==0], tumor_locs_post[kmeans_post[metric].labels_==1]],
            [tumor_grade_post[kmeans_post[metric].labels_==0], tumor_grade_post[kmeans_post[metric].labels_==1]],
            [tumor_ventricles_post[kmeans_post[metric].labels_==0], tumor_ventricles_post[kmeans_post[metric].labels_==1]]
            ])
        summary.loc[len(summary.index)] = ["Size", groups_post[metric][0][0].mean(), groups_post[metric][0][1].mean(), groups_post[metric][0][0].std()/np.sqrt(len(groups_post[metric][0][0])), groups_post[metric][0][1].std()/np.sqrt(len(groups_post[metric][0][1]))]
        summary.loc[len(summary.index)] = ["Histology", groups_post[metric][1][0].mean(), groups_post[metric][1][1].mean(), groups_post[metric][1][0].std()/np.sqrt(len(groups_post[metric][1][0])), groups_post[metric][1][1].std()/np.sqrt(len(groups_post[metric][1][1]))]
        summary.loc[len(summary.index)] = ["Location", groups_post[metric][2][0].mean(), groups_post[metric][2][1].mean(), groups_post[metric][2][0].std()/np.sqrt(len(groups_post[metric][2][0])), groups_post[metric][2][1].std()/np.sqrt(len(groups_post[metric][2][1]))]
        summary.loc[len(summary.index)] = ["Grade", groups_post[metric][3][0].mean(), groups_post[metric][3][1].mean(), groups_post[metric][3][0].std()/np.sqrt(len(groups_post[metric][3][0])), groups_post[metric][3][1].std()/np.sqrt(len(groups_post[metric][3][1]))]
        summary.loc[len(summary.index)] = ["Ventricles", groups_post[metric][4][0].mean(), groups_post[metric][4][1].mean(), groups_post[metric][4][0].std()/np.sqrt(len(groups_post[metric][4][0])), groups_post[metric][4][1].std()/np.sqrt(len(groups_post[metric][4][1]))]
        
        for cluster in range(n_clusters):            
            ax[metric].scatter(
                Embeded_post[metric,:,:][kmeans_post[metric].labels_==cluster,0], Embeded_post[metric,:,:][kmeans_post[metric].labels_==cluster,1],
                color=colors[metric,cluster,:]
                )
        ax[metric].set_title(
            'Metric: {}'.format(metrics[metric]),
            fontsize=14)
        summary.to_csv(numerical_path+'metric-{}_ses-postop.tsv'.format(metrics[metric]), sep='\t', index=False)
    plt.savefig(figures_path+'clusters_ses-postop.png', dpi=100) """

