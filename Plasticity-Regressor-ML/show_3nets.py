import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.graphs import GraphFromTensor, GraphFromCSV
from utils.data import delete_rois

try:
    with open("./model_comparisson.txt", 'r') as f:
        all_models = f.read().splitlines()
except:
    raise ValueError("No model_comparisson.txt file found")

# The order matters
model1 = all_models[0] #'results_fcnet_mse' #FCNET
model2 = all_models[1] #'results_huber_linear' #HUBER
model3 = all_models[2] #'results_null_mse' #NULL

# Load graphs
i, j, k = 3, 15, 28
mlp_sub1 = f'{model1}/predictions/CMs/sub-PAT0{str(i)}_ses-postop_prediction.csv'
mlp_sub2 = f'{model1}/predictions/CMs/sub-PAT{str(j)}_ses-postop_prediction.csv'
mlp_sub3 = f'{model1}/predictions/CMs/sub-PAT{str(k)}_ses-postop_prediction.csv'
hub_sub1 = f'{model2}/predictions/CMs/sub-PAT0{str(i)}_ses-postop_prediction.csv'
hub_sub2 = f'{model2}/predictions/CMs/sub-PAT{str(j)}_ses-postop_prediction.csv'
hub_sub3 = f'{model2}/predictions/CMs/sub-PAT{str(k)}_ses-postop_prediction.csv'
nul_sub1 = f'{model3}/predictions/CMs/sub-PAT0{str(i)}_ses-postop_prediction.csv'
nul_sub2 = f'{model3}/predictions/CMs/sub-PAT{str(j)}_ses-postop_prediction.csv'
nul_sub3 = f'{model3}/predictions/CMs/sub-PAT{str(k)}_ses-postop_prediction.csv'
g_truth1 = f'../Data/structural/graphs/hybrid/ses-postop/sub-PAT0{str(i)}_ses-postop_flatCM.csv'
g_truth2 = f'../Data/structural/graphs/hybrid/ses-postop/sub-PAT{str(j)}_ses-postop_flatCM.csv'
g_truth3 = f'../Data/structural/graphs/hybrid/ses-postop/sub-PAT{str(k)}_ses-postop_flatCM.csv'

mlp_sub1 = np.array(pd.read_csv(mlp_sub1, sep=',', header=None))
mlp_sub2 = np.array(pd.read_csv(mlp_sub2, sep=',', header=None))
mlp_sub3 = np.array(pd.read_csv(mlp_sub3, sep=',', header=None))
hub_sub1 = np.array(pd.read_csv(hub_sub1, sep=',', header=None))
hub_sub2 = np.array(pd.read_csv(hub_sub2, sep=',', header=None))
hub_sub3 = np.array(pd.read_csv(hub_sub3, sep=',', header=None))
nul_sub1 = np.array(pd.read_csv(nul_sub1, sep=',', header=None))
nul_sub2 = np.array(pd.read_csv(nul_sub2, sep=',', header=None))
nul_sub3 = np.array(pd.read_csv(nul_sub3, sep=',', header=None))

# Display some properties
print("FCNET:")
print('% of negative connections below 0:')
print(f"sub-PAT0{str(i)}: {np.sum(mlp_sub1<-0.0001)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(mlp_sub2<-0.0001)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(mlp_sub3<-0.0001)*100/(166*165/2)} %")
print('% of negative connections below -0.5:')
print(f"sub-PAT0{str(i)}: {np.sum(mlp_sub1<-0.5)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(mlp_sub2<-0.5)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(mlp_sub3<-0.5)*100/(166*165/2)} %")
print('% of negative connections below -1:')
print(f"sub-PAT0{str(i)}: {np.sum(mlp_sub1<-1)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(mlp_sub2<-1)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(mlp_sub3<-1)*100/(166*165/2)} %")
print("============================================")
print("HUBER:")
print('% of negative connections below 0:')
print(f"sub-PAT0{str(i)}: {np.sum(hub_sub1<-0.0001)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(hub_sub2<-0.0001)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(hub_sub3<-0.0001)*100/(166*165/2)} %")
print('% of negative connections below -0.5:')
print(f"sub-PAT0{str(i)}: {np.sum(hub_sub1<-0.5)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(hub_sub2<-0.5)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(hub_sub3<-0.5)*100/(166*165/2)} %")
print('% of negative connections below -1:')
print(f"sub-PAT0{str(i)}: {np.sum(hub_sub1<-1)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(hub_sub2<-1)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(hub_sub3<-1)*100/(166*165/2)} %")
print("NULL:")
print('% of negative connections below 0:')
print(f"sub-PAT0{str(i)}: {np.sum(nul_sub1<-0.0001)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(nul_sub2<-0.0001)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(nul_sub3<-0.0001)*100/(166*165/2)} %")
print('% of negative connections below -0.5:')
print(f"sub-PAT0{str(i)}: {np.sum(nul_sub1<-0.5)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(nul_sub2<-0.5)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(nul_sub3<-0.5)*100/(166*165/2)} %")
print('% of negative connections below -1:')
print(f"sub-PAT0{str(i)}: {np.sum(nul_sub1<-1)*100/(166*165/2)} %")
print(f"sub-PAT{str(j)}: {np.sum(nul_sub2<-1)*100/(166*165/2)} %")
print(f"sub-PAT{str(k)}: {np.sum(nul_sub3<-1)*100/(166*165/2)} %")

# Reshuffle FCNET graphs
sg = GraphFromTensor(mlp_sub1, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
mlp_sub1 = sg.get_connections()
sg = GraphFromTensor(mlp_sub2, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
mlp_sub2 = sg.get_connections()
sg = GraphFromTensor(mlp_sub3, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
mlp_sub3 = sg.get_connections()

# Reshuflle HUBER graphs
sg = GraphFromTensor(hub_sub1, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
hub_sub1 = sg.get_connections()
sg = GraphFromTensor(hub_sub2, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
hub_sub2 = sg.get_connections()
sg = GraphFromTensor(hub_sub3, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
hub_sub3 = sg.get_connections()

# Reshuffle NULL graphs
sg = GraphFromTensor(nul_sub1, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
nul_sub1 = sg.get_connections()
sg = GraphFromTensor(nul_sub2, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
nul_sub2 = sg.get_connections()
sg = GraphFromTensor(nul_sub3, '', rois=166)
sg.process_graph(log=False, reshuffle=True, save=False, show=False)
nul_sub3 = sg.get_connections()

# Reshuffle ground truth
sg = GraphFromCSV(g_truth1, '', rois=170)
g_truth1 = torch.tensor(sg.unflatten_graph(to_default=False, save_flat=False)).unsqueeze(0)
sg = GraphFromCSV(g_truth2, '', rois=170)
g_truth2 = torch.tensor(sg.unflatten_graph(to_default=False, save_flat=False)).unsqueeze(0)
sg = GraphFromCSV(g_truth3, '', rois=170)
g_truth3 = torch.tensor(sg.unflatten_graph(to_default=False, save_flat=False)).unsqueeze(0)
g_truth, _ = delete_rois(torch.cat((g_truth1, g_truth2, g_truth3), dim=0),ROIs=[35,36,81,82])

sg = GraphFromTensor(g_truth[0], '', rois=166)
sg.process_graph(log=True, reshuffle=True, save=False, show=False)
g_truth1 = sg.get_connections()
sg = GraphFromTensor(g_truth[1], '', rois=166)
sg.process_graph(log=True, reshuffle=True, save=False, show=False)
g_truth2 = sg.get_connections()
sg = GraphFromTensor(g_truth[2], '', rois=166)
sg.process_graph(log=True, reshuffle=True, save=False, show=False)
g_truth3 = sg.get_connections()

# Residuals
res_mlp_sub1 = np.abs(mlp_sub1 - g_truth1)
res_mlp_sub2 = np.abs(mlp_sub2 - g_truth2)
res_mlp_sub3 = np.abs(mlp_sub3 - g_truth3)
res_hub_sub1 = np.abs(hub_sub1 - g_truth1)
res_hub_sub2 = np.abs(hub_sub2 - g_truth2)
res_hub_sub3 = np.abs(hub_sub3 - g_truth3)
res_nul_sub1 = np.abs(nul_sub1 - g_truth1)
res_nul_sub2 = np.abs(nul_sub2 - g_truth2)
res_nul_sub3 = np.abs(nul_sub3 - g_truth3)

########### FIGURE MLP ###########
cmin, cmax = 0, 10.5
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(8,14))
plt.subplots_adjust(left=0.05,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.95, 
                    wspace=0.01, 
                    hspace=0.01)

one = ax1.imshow(g_truth1, vmin=cmin, vmax=cmax)
ax1.set_ylabel('sub-PAT0'+str(i), fontsize=16)
ax2.imshow(mlp_sub1, vmin=cmin, vmax=cmax)
ax3.imshow(res_mlp_sub1, vmin=cmin, vmax=cmax)

ax4.imshow(g_truth2, vmin=cmin, vmax=cmax)
ax4.set_ylabel('sub-PAT'+str(j), fontsize=16)
ax5.imshow(mlp_sub2, vmin=cmin, vmax=cmax)
ax6.imshow(res_mlp_sub2, vmin=cmin, vmax=cmax)

ax7.imshow(g_truth3, vmin=cmin, vmax=cmax)
ax7.set_xlabel('Ground truth', fontsize=16), ax7.set_ylabel('sub-PAT'+str(k), fontsize=16)
ax8.imshow(mlp_sub3, vmin=cmin, vmax=cmax)
ax8.set_xlabel('Prediction', fontsize=16)
ax9.imshow(res_mlp_sub3, vmin=cmin, vmax=cmax)
ax9.set_xlabel('Residuals', fontsize=16)

ax1.set_xticks([]),ax1.set_yticks([])
ax2.set_xticks([]),ax2.set_yticks([])
ax3.set_xticks([]),ax3.set_yticks([])
ax4.set_xticks([]),ax4.set_yticks([])
ax5.set_xticks([]),ax5.set_yticks([])
ax6.set_xticks([]),ax6.set_yticks([])
ax7.set_xticks([]),ax7.set_yticks([])
ax8.set_xticks([]),ax8.set_yticks([])
ax9.set_xticks([]),ax9.set_yticks([])

cbar_ax = fig.add_axes([ 0.05, 0.9, 0.94, 0.05])
cbar = fig.colorbar(one, cax=cbar_ax, orientation='horizontal', ticklocation='top')
cbar.set_label(label='Connection Strength', fontsize=16, weight='bold')
cbar.ax.tick_params(labelsize=12)

plt.savefig(f'{model1}/mlp_predictions.eps', dpi=1000)
plt.savefig(f'{model1}/mlp_predictions.svg', dpi=1000)

########### FIGURE HUBER ###########
cmin, cmax = 0, 10.5
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(8,14))
plt.subplots_adjust(left=0.05,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.95, 
                    wspace=0.01, 
                    hspace=0.01)

one = ax1.imshow(g_truth1, vmin=cmin, vmax=cmax)
ax1.set_ylabel('sub-PAT0'+str(i), fontsize=16)
ax2.imshow(hub_sub1, vmin=cmin, vmax=cmax)
ax3.imshow(res_hub_sub1, vmin=cmin, vmax=cmax)

ax4.imshow(g_truth2, vmin=cmin, vmax=cmax)
ax4.set_ylabel('sub-PAT'+str(j), fontsize=16)
ax5.imshow(hub_sub2, vmin=cmin, vmax=cmax)
ax6.imshow(res_hub_sub2, vmin=cmin, vmax=cmax)

ax7.imshow(g_truth3, vmin=cmin, vmax=cmax)
ax7.set_xlabel('Ground truth', fontsize=16), ax7.set_ylabel('sub-PAT'+str(k), fontsize=16)
ax8.imshow(hub_sub3, vmin=cmin, vmax=cmax)
ax8.set_xlabel('Prediction', fontsize=16)
ax9.imshow(res_hub_sub3, vmin=cmin, vmax=cmax)
ax9.set_xlabel('Residuals', fontsize=16)

ax1.set_xticks([]),ax1.set_yticks([])
ax2.set_xticks([]),ax2.set_yticks([])
ax3.set_xticks([]),ax3.set_yticks([])
ax4.set_xticks([]),ax4.set_yticks([])
ax5.set_xticks([]),ax5.set_yticks([])
ax6.set_xticks([]),ax6.set_yticks([])
ax7.set_xticks([]),ax7.set_yticks([])
ax8.set_xticks([]),ax8.set_yticks([])
ax9.set_xticks([]),ax9.set_yticks([])

cbar_ax = fig.add_axes([ 0.05, 0.9, 0.94, 0.05])
cbar = fig.colorbar(one, cax=cbar_ax, orientation='horizontal', ticklocation='top')
cbar.set_label(label='Connection Strength', fontsize=16, weight='bold')
cbar.ax.tick_params(labelsize=12)

plt.savefig(f'{model2}/huber_predictions.eps', dpi=1000)
plt.savefig(f'{model2}/huber_predictions.svg', dpi=1000)