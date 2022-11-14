import pandas as pd
import numpy as np
import scipy.stats
import itertools

#############################################
### Loading results from specified models ###
#############################################
try:
    with open("./model_comparisson.txt", 'r') as f:
        all_models = {}
        names = []
        for i in f.read().splitlines():
            name = f'./{i}/'+'_'.join(i.split('_')[1:]) + '_LOO-testing.tsv'
            names.append(i)
            model = pd.read_csv(name, sep='\t', index_col=False)
            model.drop(columns=['Subject'], inplace=True)
            for metric in model.columns:
                try:
                    all_models[metric].append(np.array(model[metric]))
                except:
                    all_models[metric] = [np.array(model[metric])]

except:
    raise ValueError("No model_comparisson.txt file found")

##################################
### Statistics for each metric ###
##################################
for metric in all_models:
    print("==============================")
    print("Metric:", metric)
    print("------------------------------")
    array_metric = np.array(all_models[metric])
    means, std = np.mean(array_metric, axis=1), np.std(array_metric, axis=1)
    sem = std/np.sqrt(array_metric.shape[1])
    _, p_ANOVA = scipy.stats.f_oneway(*all_models[metric])
    _, p_Kruskal = scipy.stats.kruskal(*all_models[metric])
    print("ANOVA p-value:", p_ANOVA)
    print("Kruskal p-value:", p_Kruskal)
    print("------------------------------")
    combs = list(itertools.combinations(all_models[metric], 2))
    p_T, p_U = [], []
    for pairs in combs:
        # Testing for one sided hypothesis ==> p-value/2
        p1 = scipy.stats.ttest_ind(pairs[0], pairs[1])[1]/2
        p2 = scipy.stats.mannwhitneyu(pairs[0], pairs[1])[1]/2
        p_T.append(p1)
        p_U.append(p2)
        print(f"mean 1: {np.mean(pairs[0])} vs mean 2: {np.mean(pairs[1])}")
        print(f"one sided T-test p-value: {p1}")
        print(f"one sided U-test p-value: {p2}")
    print("==============================\n")
