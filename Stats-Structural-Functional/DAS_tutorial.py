import numpy as np
import matplotlib.pylab as plt

# Preparing signals
points = 200
t = np.linspace(0, 4*np.pi, points)
omega1, phase1 = 1, 0
omega2, phase2 = 1, np.pi
omega3, phase3 = 8, 5*np.pi
z1 = np.sin(omega1*t+phase1)
z2 = np.sin(omega2*t+phase2)
z3 = np.sin(omega3*t+phase3)
nps = 10
dp = int(100/nps)
cutoffs = np.zeros((3,nps+1), dtype=np.int64)

# Fourier reconstruction z1 
bin_power, Power, Total_P = np.zeros((3,nps+1)), np.zeros((3,nps+1)), np.zeros((3,))
fft = np.array([np.fft.rfft(z1, n=points), np.fft.rfft(z2, n=points), np.fft.rfft(z3, n=points)])
for s in range(3):
    lower_cut = 0
    for j, percent in enumerate(range(0, 100+dp, dp)):
        cutoff = int(len(fft[s,:])*percent/100)
        cutoffs[s,j] = percent
        fft_cut = fft[s,:cutoff]
        bin_power[s,j] = np.sum(np.square(np.abs(fft[s,lower_cut:cutoff])))
        Power[s,j] = np.sum(np.square(np.abs(fft_cut)))
        lower_cut = cutoff
    Total_P[s] = Power[s,-1]
    bin_power[s,:] = 100*bin_power[s,:]/Power[s,-1]
    Power[s,:] = 100*Power[s,:]/Power[s,-1]

fig, ax = plt.subplots(2,2, figsize=(10,8))
plt.subplots_adjust(left=0.04,
                bottom=0.07, 
                right=0.975, 
                top=0.95, 
                wspace=0.2, 
                hspace=0.15)
ax = ax.flatten()
plt.suptitle("z$_i$(t) = sin($\omega_i$t+$\phi_i$)", fontsize=15, fontweight='bold') 
plt.gcf().text(0.01, 0.95, "A", fontsize=20, fontweight="bold")
plt.gcf().text(0.01, 0.48, "B", fontsize=20, fontweight="bold")

ax[0].plot(t, z1, color='blue', linewidth=2, label=f'$\omega_1$={omega1} $\phi_1$={phase1}')
ax[0].plot(t, z2, color='red', linewidth=2, label=f'$\omega_2$={omega2} $\phi_2$=$\pi$')
ax[0].plot(t, 0.5*(z1+z2), color='black', linewidth=2.5, label=f'1/2 * ($z_1$+$z_2$)')
ax[0].spines['right'].set_visible(False), ax[0].spines['top'].set_visible(False), ax[0].spines['bottom'].set_visible(False)
ax[0].set_xlim([0,t[-1]+0.1]), ax[0].set_xlabel("time (s)")
ax[0].set_ylim([-1.1,1.5]), ax[0].xaxis.set_ticks_position('none')
ax[0].set_yticks([-1,0,1]), ax[0].set_yticklabels(["-1","0","1"])
ax[0].legend(frameon=False, loc='upper left', ncol=3)

ax[1].bar(cutoffs[0], bin_power[0,:], width=4, alpha=0.5, color='blue', label=f"P$_T$(z$_1$) = {int(Total_P[0])}")
ax[1].bar(cutoffs[1], bin_power[1,:], width=4, alpha=0.5, color='red', label=f"P$_T$(z$_2$) = {int(Total_P[1])}")
ax[1].plot(cutoffs[0], Power[0,:], '-', linewidth=2, alpha=0.5, color='blue')
ax[1].plot(cutoffs[1], Power[1,:], '-', linewidth=2, alpha=0.5, color='red')
ax[1].fill_between(cutoffs[1], Power[0,:], Power[1,:], color='green', alpha=0.4, label=f'DAS(z$_1$,z$_2$) = {round(np.trapz(Power[0,:]-Power[1,:],cutoffs[0]))}')
ax[1].legend(frameon=False)
ax[1].spines['right'].set_visible(False), ax[1].spines['top'].set_visible(False)
ax[1].set_xticks([0,10,20,40,60,80,100]), ax[1].set_xticklabels(['0','10','20','40','60','80','100'])
ax[1].set_yticks([0,20,40,60,80,100]), ax[1].set_yticklabels(['0','20','40','60','80','100'])
ax[1].set_xlabel("$\omega_c$ (%)", fontsize=12), ax[1].set_ylabel("$P_{\omega_c}$ (%)", fontsize=12, labelpad=0)


ax[2].plot(t, z1, color='blue', linewidth=2, label=f'$\omega_1$={omega1} $\phi_1$={phase1}')
ax[2].plot(t, z3, color='orange', linewidth=2, label=f'$\omega_3$={omega3} $\phi_3$=5$\pi$')
ax[2].plot(t, 0.5*(z1+z3), color='black', linewidth=2.5, label=f'1/2 * ($z_1$+$z_3$)')
ax[2].spines['right'].set_visible(False), ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
ax[2].set_xlim([0,t[-1]+0.1]), ax[2].set_xlabel("time (s)")
ax[2].set_ylim([-1.1,1.5]), ax[2].xaxis.set_ticks_position('none')
ax[2].set_yticks([-1,0,1]), ax[2].set_yticklabels(["-1","0","1"])
ax[2].legend(frameon=False, loc='upper left', ncol=3)

ax[3].bar(cutoffs[0], bin_power[0,:], width=4, color='blue', label=f"P$_T$(z$_1$) = {int(Total_P[0])}")
ax[3].bar(cutoffs[2], bin_power[2,:], width=4, color='orange', label=f"P$_T$(z$_3$) = {int(Total_P[2])}")
ax[3].plot(cutoffs[0], Power[0,:], '-', linewidth=2, alpha=0.5, color='blue')
ax[3].plot(cutoffs[2], Power[2,:], '-', linewidth=2, alpha=0.5, color='orange')
ax[3].fill_between(cutoffs[2], Power[0,:], Power[2,:], color='green', alpha=0.4, label=f'DAS(z$_1$,z$_3$) = {round(np.trapz(Power[0,:]-Power[2,:],cutoffs[0]))}')
ax[3].legend(frameon=False)
ax[3].spines['right'].set_visible(False), ax[3].spines['top'].set_visible(False)
ax[3].set_xticks([0,10,20,40,60,80,100]), ax[3].set_xticklabels(['0','10','20','40','60','80','100'])
ax[3].set_yticks([0,20,40,60,80,100]), ax[3].set_yticklabels(['0','20','40','60','80','100'])
ax[3].set_xlabel("$\omega_c$ (%)", fontsize=12), ax[3].set_ylabel("$P_{\omega_c}$ (%)", fontsize=12, labelpad=0)

plt.savefig("./RESULTS/figures/DAS_tutorial.svg", dpi=1000)
