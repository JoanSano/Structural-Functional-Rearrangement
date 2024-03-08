import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
from pathlib import Path
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import logging, sys

from models.methods import Model, return_specs, to_array
from models.networks import LinearRegres, NonLinearRegres
from models.metrics import degree_distribution, PCC, BayesianWeightedLoss, CosineSimilarity, KL_JS_divergences
from utils.data import check_path, graph_dumper, two_session_graph_loader, prepare_data
from utils.graphs import GraphFromCSV, GraphFromTensor, create_anat_prior, load_anat_prior
from utils.paths import get_subjects, get_info

parser = argparse.ArgumentParser()
# General settings
parser.add_argument('--mode', type=str, default='train', choices=['train', 'stats'], help="Train the model or just report statistics")
parser.add_argument('-D', '--device', type=str, default='cuda', help="Device in which to run the code")
parser.add_argument('-F', '--folder', type=str, default='results', help="Results directory")
parser.add_argument('-M', '--model', type=str, default='linear', help="Trained model name")
parser.add_argument('-W', '--wandb', action=argparse.BooleanOptionalAction, help="Whether to use wandb")
parser.add_argument('--null_model', action=argparse.BooleanOptionalAction, help="Whether not to train the model to obtain a benchmark")
parser.add_argument('--tractography', type=str, default='msmt', choices=["msmt", "hybrid"], help="Whether not to train the model to obtain a benchmark")

# Data specs
parser.add_argument('-S', '--split', type=int, default=20, help="Training and testing splitting")
parser.add_argument('-R', '--rois', type=int, default=166, help="Number of ROIs to use")
parser.add_argument('-A', '--augment', type=int, default=1, help="Data augmentation factor")
parser.add_argument('-V', '--validation', action=argparse.BooleanOptionalAction, help="Add validation step")
parser.add_argument('-P', '--prior', action=argparse.BooleanOptionalAction, help="Load available prior")
parser.add_argument('-T', '--threshold', type=float, default=0.2, help="Threshold for creating the prior")

# Machine-learning specs
parser.add_argument('-E', '--epochs', type=int, default=10, help="Number of epochs to train")
parser.add_argument('-LR', '--learning_rate', type=float, default=0.01, help="Learning Rate")
parser.add_argument('-O', '--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'], help="Optimizer")
parser.add_argument('--val_freq', type=int, default=5, help="Number of epochs between validation steps")
parser.add_argument('-B', '--batch', type=int, default=4, help="Batch size")
parser.add_argument('-RE', '--regressor', type=str, default='linear', choices=['linear','nonlinear'], help="Type of regression")
parser.add_argument('-L', '--loss', type=str, default='mse', choices=['mse', 'huber'], help="Reconstruction loss used in training")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

if __name__ == '__main__':
    # Relevant paths
    folder = args.folder+'_'+args.model+'/'
    check_path(folder)
    CMs_path = check_path(folder+'predictions/CMs/')
    figs_path = check_path(folder+'figures/')
    
    if args.mode == 'train':
        # Dump configuration file for reproducibility
        with open(folder+'command_log.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        # Specifing TORCH device to use
        logging.info(torch.cuda.is_available())
        if args.device == 'cuda' and torch.cuda.is_available():
            args.device == torch.device('cuda')
            logging.info(" Training will be done using gpu")
        else: 
            args.device = torch.device('cpu')
            logging.info(" Training will be done using cpu")
        #args.device = 'cpu'
        # Preparing data
        logging.info(" Loading data ...")
        (CONTROL, CON_subjects), (data, PAT_subjects), (PAT_1session, PAT_1session_subjects) = prepare_data(
            f'../Data/structural/graphs/{args.tractography}/', dtype=torch.float64, rois=170, norm=False, flatten=True, del_rois=[35,36,81,82]
        )

        # Creating or loading priors         
        if args.prior:
            logging.info(" Loading prior ...")
            prior, mean_connections = load_anat_prior(folder)
        else:
            logging.info(" Creating prior ...")
            prior, mean_connections = create_anat_prior(CONTROL, folder, save=True, threshold=args.threshold)
            sg = GraphFromCSV(folder+'/prior.csv', 'prior', folder, rois=args.rois)
            sg.unflatten_graph(to_default=True, save_flat=True)
            sg.process_graph(log=False, reshuffle=True, bar_label='Probability of Connection')

        # Cross-Validation
        CV = LeaveOneOut()
        N_folds = CV.get_n_splits(data[0])

        logging.info(" All OK!")
        ################
        ### Training ###
        ################

        # Results
        CV_summary = pd.DataFrame(columns=['Subject', 'MSE', 'MAE', 'PCC', 'CosineSimilarity', 'KL_Div', 'JS_Div'])
        Degree = pd.DataFrame(columns=['Subject', 'Predicted', 'Ground Truth', 'Degrees'])
        final_regres, _, _ = return_specs(args, prior=prior)
        final_model = final_regres.state_dict()

        for fold, (train_index, test_index) in enumerate(CV.split(data[0])):
            input_train, input_test = data[0][train_index], data[0][test_index].to(args.device)
            target_train, target_test = data[1][train_index], data[1][test_index]
            data_fold = (input_train, target_train)
            subject = PAT_subjects[test_index[0]]

            # Defining the model
            regres, loss, optimizer = return_specs(args, prior=prior.to(args.device))
            model = Model(regres, optimizer, loss, data_fold, args)

            # Training and testing
            if not args.null_model:# or args.null_model==False:
                model.train(fold, N_folds)
            else:
                logging.info(" ===== NULL: Fold {}/{} ======".format(fold+1, N_folds))
            pred_LOO = model.test(input_test, prior=prior).cpu()
            
            sg = GraphFromTensor(pred_LOO, subject+'_ses-postop_prediction', base_dir=CMs_path, rois=args.rois)
            sg.unflatten_graph(to_default=True, save_flat=True)
            sg.save()
            sg.process_graph(log=False, reshuffle=True)

            # Metrics
            test_mse = F.mse_loss(pred_LOO, target_test)
            test_mae = F.l1_loss(pred_LOO, target_test)
            _, pcc, _ = PCC().forward(pred_LOO, target_test)
            _, cs, _ = CosineSimilarity().forward(pred_LOO, target_test)
            kl_div, js_div = KL_JS_divergences(pred_LOO, target_test, rois=args.rois)
            CV_summary.loc[len(CV_summary.index)] = [subject, test_mse.item(), test_mae.item(), pcc.item(), cs.item(), kl_div.item(), js_div.item()]
            dist, dgs = degree_distribution(pred_LOO, args.rois)
            Degree.loc[len(Degree.index)] = [subject, dist, degree_distribution(target_test, args.rois)[0], dgs]

            # Creating the final model
            for key in final_model.keys():
                if fold==0:
                    final_model[key] = regres.state_dict()[key]
                else:
                    final_model[key] += regres.state_dict()[key]/N_folds
        
        # Saving the mean model
        final_regres.load_state_dict(final_model)
        torch.save(final_regres, folder+args.model+'.ckpt') 

        # Saving performance evaluation
        CV_summary.to_csv(folder+args.model+'_LOO-testing.tsv', sep='\t', index=False)
        Degree.to_csv(folder+args.model+'_degree_distribution.tsv', sep='\t', index=False)

        # TODO: Save the 1-fold predictions (flattened, unshuffled) + the connectivity matrice? (add deleted rois)

    if args.mode == 'stats':
        try:
            CV_summary = pd.read_csv(folder+args.model+'_LOO-testing.tsv', sep='\t')
        except:
            raise ValueError('No CV summary found. Run with --train')

        # Reading-loading results
        PAT_subjects = list(CV_summary['Subject'].values)
        mse = np.array(CV_summary['MSE'], dtype=np.float64)
        mae = np.array(CV_summary['MAE'], dtype=np.float64)
        pcc = np.array(CV_summary['PCC'], dtype=np.float64)
        cs = np.array(CV_summary['CosineSimilarity'], dtype=np.float64)
        kl = np.array(CV_summary['KL_Div'], dtype=np.float64)
        js = np.array(CV_summary['JS_Div'], dtype=np.float64)
        N_folds = len(PAT_subjects)

        # Loading patient information and REORDERING in the same way as CV_summary!!!
        info = pd.read_csv('../Data/participants.tsv', sep='\t')
        info = info[info["participant_id"].str.contains("CON") == False]
        info.set_index(info.participant_id, inplace=True)
        info.drop(['participant_id'], axis=1, inplace=True)
        info.index.name = None
        tumor_sizes = {k: dict(info["tumor size (cub cm)"])[k] for k in PAT_subjects}
        tumor_types = {k: dict(info["tumor type & grade"])[k] for k in PAT_subjects}
        tumor_locs = {k: dict(info["tumor location"])[k] for k in PAT_subjects}
        tumor_ventricles = {k: dict(info["ventricles"])[k] for k in PAT_subjects}       
        tumor_grade = {k: dict(info["tumor type & grade"])[k] for k in PAT_subjects}
        # Acces data by converting to list and to numpy array np.array(list(values), dtype)

        #################################################################################################
        ### Mean and Standard Error of the Mean of the current model and metric ==> E +/- STD/SQRT(N) ###
        #################################################################################################
        mse_mean, mse_std= np.mean(mse), np.std(mse)
        mae_mean, mae_std = np.mean(mae), np.std(mae)
        pcc_mean, pcc_std = np.mean(pcc), np.std(pcc)
        cs_mean, cs_std = np.mean(cs), np.std(cs)
        kl_mean, kl_std = np.mean(kl), np.std(kl)
        js_mean, js_std = np.mean(js), np.std(js)
        CV_summary.loc[len(CV_summary.index)] = [
            'Mean +/- SEM', 
            str(round(mse_mean.item(),4))+' +/- '+str(round(mse_std/np.sqrt(N_folds),4)), 
            str(round(mae_mean.item(),4))+' +/- '+str(round(mae_std/np.sqrt(N_folds),4)),
            str(round(pcc_mean.item(),4))+' +/- '+str(round(pcc_std/np.sqrt(N_folds),4)),
            str(round(cs_mean.item(),4))+' +/- '+str(round(cs_std/np.sqrt(N_folds),4)),
            str(round(kl_mean.item(),4))+' +/- '+str(round(kl_std/np.sqrt(N_folds),4)),
            str(round(js_mean.item(),4))+' +/- '+str(round(js_std/np.sqrt(N_folds),4))
            ]

        ######################################################
        ### Testing for an outlier in the Cross Validation ###
        ######################################################
        # 1) z-scores
        from scipy.stats import zscore
        mse_z, mae_z, pcc_z, cs_z, kl_z, js_z = zscore(mse), zscore(mae), zscore(pcc), zscore(cs), zscore(kl), zscore(js)
        one_sigma = [ # % inside 1 sigma
            np.sum((mse_z>-1)*(mse_z<1))/N_folds * 100,
            np.sum((mae_z>-1)*(mae_z<1))/N_folds * 100,
            np.sum((pcc_z>-1)*(pcc_z<1))/N_folds * 100,
            np.sum((cs_z>-1)*(cs_z<1))/N_folds * 100,
            np.sum((kl_z>-1)*(kl_z<1))/N_folds * 100,
            np.sum((js_z>-1)*(js_z<1))/N_folds * 100
        ]
        two_sigma = [ # % 2 sigmas
            np.sum((mse_z>-2)*(mse_z<2))/N_folds * 100,
            np.sum((mae_z>-2)*(mae_z<2))/N_folds * 100,
            np.sum((pcc_z>-2)*(pcc_z<2))/N_folds * 100,
            np.sum((cs_z>-2)*(cs_z<2))/N_folds * 100,
            np.sum((kl_z>-2)*(kl_z<2))/N_folds * 100,
            np.sum((js_z>-2)*(js_z<2))/N_folds * 100
        ]
        CV_summary.loc[len(CV_summary.index)] = ['1-sigma (%)', *one_sigma]
        CV_summary.loc[len(CV_summary.index)] = ['2-sigma (%)', *two_sigma]
        CV_summary.loc[len(CV_summary.index)] = [
            'Z-score outlier', 
            PAT_subjects[np.argmax(abs(mse_z))]+' z='+str(round(np.max(abs(mse_z)),4)), 
            PAT_subjects[np.argmax(abs(mae_z))]+' z='+str(round(np.max(abs(mae_z)),4)), 
            PAT_subjects[np.argmax(abs(pcc_z))]+' z='+str(round(np.max(abs(pcc_z)),4)), 
            PAT_subjects[np.argmax(abs(cs_z))]+' z='+str(round(np.max(abs(cs_z)),4)),
            PAT_subjects[np.argmax(abs(kl_z))]+' z='+str(round(np.max(abs(kl_z)),4)),
            PAT_subjects[np.argmax(abs(js_z))]+' z='+str(round(np.max(abs(js_z)),4))
            ]
        
        # 2) Grubb's test for outlier
        from models.methods import grubbs_test
        grubbs_results = ['Grubbs 2-sided']
        h,p,out = grubbs_test(mse)
        if h:
            grubbs_results.append(PAT_subjects[out]+' p='+str(round(p,4)))
        else:
            grubbs_results.append('No outlier p='+str(round(p,4)))
        h,p,out = grubbs_test(mae)
        if h:
            grubbs_results.append(PAT_subjects[out]+' p='+str(round(p,4)))
        else:
            grubbs_results.append('No outlier p='+str(round(p,4)))
        h,p,out = grubbs_test(pcc)
        if h:
            grubbs_results.append(PAT_subjects[out]+' p='+str(round(p,4)))
        else:
            grubbs_results.append('No outlier p='+str(round(p,4)))
        h,p,out = grubbs_test(cs)
        if h:
            grubbs_results.append(PAT_subjects[out]+' p='+str(round(p,4)))
        else:
            grubbs_results.append('No outlier p='+str(round(p,4)))
        h,p,out = grubbs_test(kl)
        if h:
            grubbs_results.append(PAT_subjects[out]+' p='+str(round(p,4)))
        else:
            grubbs_results.append('No outlier p='+str(round(p,4)))
        h,p,out = grubbs_test(js)
        if h:
            grubbs_results.append(PAT_subjects[out]+' p='+str(round(p,4)))
        else:
            grubbs_results.append('No outlier p='+str(round(p,4)))
        CV_summary.loc[len(CV_summary.index)] = grubbs_results

        # 3) Cyclical T-/Wilcoxon-test 
        from scipy.stats import ttest_1samp, wilcoxon
        p_ttest, p_wtest = np.zeros((N_folds,6)), np.zeros((N_folds,6))
        for sub in range(N_folds):
            sample_mse, pop_mse = np.delete(np.copy(mse), sub), mse[sub]
            sample_mae, pop_mae = np.delete(np.copy(mae), sub), mae[sub]
            sample_pcc, pop_pcc = np.delete(np.copy(pcc), sub), pcc[sub]
            sample_cs, pop_cs = np.delete(np.copy(cs), sub), cs[sub]
            sample_kl, pop_kl = np.delete(np.copy(kl), sub), kl[sub]
            sample_js, pop_js = np.delete(np.copy(js), sub), js[sub]

            # 1-sample 2-sided T test
            _, pt_mse = ttest_1samp(sample_mse, pop_mse)
            _, pt_mae = ttest_1samp(sample_mae, pop_mae)
            _, pt_pcc = ttest_1samp(sample_pcc, pop_pcc)
            _, pt_cs = ttest_1samp(sample_cs, pop_cs)
            _, pt_kl = ttest_1samp(sample_kl, pop_kl)
            _, pt_js = ttest_1samp(sample_js, pop_js)
            # 1-sample 2-sided Wilcoxon test
            _, pw_mse = wilcoxon(sample_mse-pop_mse)
            _, pw_mae = wilcoxon(sample_mae-pop_mae)
            _, pw_pcc = wilcoxon(sample_pcc-pop_pcc)
            _, pw_cs = wilcoxon(sample_cs-pop_cs)
            _, pw_kl = wilcoxon(sample_kl-pop_kl)
            _, pw_js = wilcoxon(sample_js-pop_js)

            CV_summary.loc[len(CV_summary.index)] = [
                PAT_subjects[sub]+' T/W-test',
                'pt='+str(round(pt_mse,4))+' pw='+str(round(pw_mse,4)),
                'pt='+str(round(pt_mae,4))+' pw='+str(round(pw_mae,4)),
                'pt='+str(round(pt_pcc,4))+' pw='+str(round(pw_pcc,4)),
                'pt='+str(round(pt_cs,4))+' pw='+str(round(pw_cs,4)),
                'pt='+str(round(pt_kl,4))+' pw='+str(round(pw_kl,4)),
                'pt='+str(round(pt_js,4))+' pw='+str(round(pw_js,4))
                ]
            p_ttest[sub,:] = np.array([pt_mse, pt_mae, pt_pcc, pt_cs, pt_kl, pt_js], dtype=np.float64)
            p_wtest[sub,:] = np.array([pw_mse, pw_mae, pw_pcc, pw_cs, pw_kl, pw_js], dtype=np.float64)
        
        CV_summary.to_csv(folder+args.model+'_stats.tsv', sep='\t', index=False)

        ###############
        ### Figures ###
        ###############
        from utils.figures import *

        """ # 1) Box plot for z-scores of the 19 folds
        boxplot(figs_path, args, mse, mse_z, mae_z, pcc_z, cs_z, kl_z, js_z, PAT_subjects)
        # 2) Normality plot for all metrics
        normality_plots(figs_path, mse_z, mae_z, pcc_z, cs_z, kl_z, js_z, args, PAT_subjects)

        
        # Correlations between metrics
        correlations = np.array([
            pearsonr(mse_z, mae_z)[0], pearsonr(mse_z, pcc_z)[0], pearsonr(mse_z, cs_z)[0], pearsonr(mae_z, pcc_z)[0], pearsonr(mae_z, cs_z)[0], pearsonr(pcc_z, cs_z)[0]
        ])
        logging.info(" Correlations between metrics:")
        logging.info(" MSE-MAE: r = " + str(correlations[0]))
        logging.info(" MSE-PCC: r = " + str(correlations[1]))
        logging.info(" MSE-CS: r = " + str(correlations[2]))
        logging.info(" MAE-PCC: r = " + str(correlations[3]))
        logging.info(" MAE-CS: r = " + str(correlations[4]))
        logging.info(" PCC-CS: r = " + str(correlations[5]))
        logging.info(" =====================================") """

        # 3) Checking for the effect of tumor size
        size_correlation(figs_path, args, mae, pcc, tumor_sizes, PAT_subjects)

        # 4) Checking for the effect of tumor type
        type_effects(figs_path, args, mae, pcc, tumor_types, PAT_subjects)

        # 5) Checking for the effect of tumor location
        location_effects(figs_path, args, mae, pcc, tumor_locs, PAT_subjects)

        # 6) Checking for the effect of periventricularity
        periventricularity_effects(figs_path, args, mae, pcc, tumor_ventricles, PAT_subjects)

        # 6) Checking for the effect of periventricularity
        grade_effects(figs_path, args, mae, pcc, tumor_grade, PAT_subjects)

        # 7) Degree distributions
        plot_degree_distribution(figs_path, args, folder+args.model+'_degree_distribution.tsv')  
