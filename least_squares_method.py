import sys
import time
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import segyio # não sendo usada
import scipy.signal
import scipy.interpolate
import pylops
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme()
print("Starting...")
start_time = time.time()

# sep = '\s+'
eoceno = pd.read_csv("Superficies/EocenoSuperior_SubvolBackground_Time.dat",
                     delim_whitespace=True, skiprows=0, usecols=(0, 1, 2), names=['X', 'Y', 'Z'])

paleoceno = pd.read_csv("Superficies/Paleoceno_SubvolBackground_Time.dat",
                        delim_whitespace=True, skiprows=0, usecols=(0, 1, 2), names=['X', 'Y', 'Z'])
print("Loading data...")
amp = np.load("Sismica/sismica_497.npy")
mback = np.load("Impedancia/lowfrequency_497.npy")
wav = np.load("Wavelets/wav_avg.npy")

xl_start = np.min(eoceno['Y'])
xl_end = np.max(eoceno['Y'])

# Seismic section of well 55
print("Plotting low frequency...")
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

c=axs[0].imshow(amp.T, aspect='auto', cmap='gray_r', vmin=-15000, vmax=15000, extent=[xl_start, xl_end, 7000, 0])
plt.colorbar(c, ax=axs[0], pad=0.01).set_label('Amplitude', size=15)

c=axs[1].imshow(mback.T, aspect='auto', cmap='viridis_r', vmin=5000, vmax=10000, extent=[xl_start, xl_end, 7000, 0])
plt.colorbar(c, ax=axs[1], pad=0.01).set_label('P-Impedance [m/s.g/cm³]', size=15)


for ax in [axs[0], axs[1]]:
    ax.grid(False)
    ax.set_ylim(3900, 2700)
    #ax.set_xlim(il_start, il_end)
    ax.set_yticks(np.arange(2700, 3901, 200))
    ax.set_yticklabels(np.arange(2700, 3901, 200), size=13)
    ax.plot(eoceno.loc[eoceno.X == 2862].Y, eoceno.loc[eoceno.X == 2862].Z,
             color='m', lw=3.0, label='Eoceno')
    ax.plot(paleoceno.loc[paleoceno.X == 2862].Y, paleoceno.loc[paleoceno.X == 2862].Z,
             color='lime', lw=3.0, label='Paleoceno')

    ax.set_ylabel("Two way time [ms]", fontsize=15)

    ax.axvline(x=5553, color='k')

axs[0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.75, 1.0), facecolor='white', edgecolor='white')
axs[0].set_xticks([])
plt.tight_layout()
plt.savefig('Figures/lowfrequency_497.png', dpi=200, bbox_inches='tight')

print("Creating wavelet...")
path = "Pocos/IP_UPS/"

df = pd.DataFrame()
names = ['Depth', 'VP', 'RHO', 'VP_4Hz', 'RHO_4Hz', 'VP_6Hz', 'RHO_6Hz',
         'VP_8Hz', 'RHO_8Hz', 'IP', 'IP_4Hz', 'IP_6Hz', 'IP_8Hz']

for file in os.listdir(path):
    well = pd.read_csv(path+file, delim_whitespace=True, skiprows=1, names=names)
    # Replacing nans
    well.replace(1e30, np.nan, inplace=True)
    # Naming well
    well['WELL'] = file.split('_')[0]
    tdr = pd.read_csv(f'Pocos/TimeVsDepth/{file.split("_")[0]}.dat', skiprows=14, delim_whitespace=True,
                      names=['MD', 'TVD', 'TVDSS', 'TWT', 'VINT'])
    interp_tdr = scipy.interpolate.interp1d(tdr['MD'], tdr['TWT'], fill_value='extrapolate', bounds_error=False)
    t_well = interp_tdr(well.Depth)

    new_samples = np.arange(2700, 3900, 4)
    interp2 = scipy.interpolate.interp1d(t_well, well.IP, bounds_error=False, fill_value='extrapolate')
    ip_ups = interp2(new_samples)
    ip_ups = scipy.ndimage.gaussian_filter(ip_ups, sigma=0.5)

    arr = np.array([new_samples, ip_ups])
    nwell = pd.DataFrame(arr.T, columns=['TWT', 'IP_UPS'])
    nwell['WELL'] = file.split('_')[0]
    nwell['IP_UPS'] = nwell['IP_UPS'].fillna(method='bfill').fillna(method='ffill')
    plt.title(nwell['WELL'].unique())
    #plt.plot(nwell['IP_UPS'])
    #plt.show()
    # Concatenate
    df = pd.concat([df, nwell], ignore_index=True)

print("Inverting data...")
amp_ls = amp.copy()
mback_ls = mback.copy()

amp_ls = amp[:, 675:975].T
mback_ls = np.log(mback[:, 675:975].T)

print("Shape of data [n_samples, n_traces]:", amp_ls.shape)

# Wavelet operator
nz, nx = amp_ls.shape
PPop = pylops.avo.poststack.PoststackLinearModelling(wav/2,
                                                     nt0=nz,
                                                     spatdims=nx,
                                                     explicit=True)

# Trace-by-trace
minv_tbt, _ = pylops.optimization.leastsquares.normal_equations_inversion(
    Op=PPop,
    y=amp_ls.ravel() / 30000,
    Regs=None,
    epsI=5e-1,
    x0=mback_ls.ravel(),
    engine='scipy',
    **dict(maxiter=1000))
minv_tbt = np.exp(mback_ls + minv_tbt.reshape(nz, nx))

img = np.tile(df.loc[df.WELL=='6-BRSA-497-ESS'].IP_UPS, (100, 1))

# Seismic section of well 55
print("Plotting final inversion...")
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

c=axs[0].imshow(amp_ls, aspect='auto', cmap='gray_r', vmin=-15000, vmax=15000, extent=[xl_start, xl_end, 3900, 2700])
plt.colorbar(c, ax=axs[0], pad=0.01).set_label('Amplitude', size=15)

c=axs[1].imshow(minv_tbt, aspect='auto', cmap='viridis_r', vmin=5000, vmax=8000, extent=[xl_start, xl_end, 3900, 2700])
plt.colorbar(c, ax=axs[1], pad=0.01).set_label('P-Impedance [m/s.g/cm³]', size=15)
axs[1].imshow(img.T, aspect='auto', cmap='viridis_r', vmin=5000, vmax=8000,
             extent=(5553-5, 5553+5, 3900, 2700), zorder=20)

for ax in [axs[0], axs[1]]:
    ax.grid(False)
    ax.set_ylim(3500, 2700)
    #ax.set_xlim(il_start, il_end)
    ax.set_yticks(np.arange(2700, 3501, 200))
    ax.set_yticklabels(np.arange(2700, 3501, 200), size=13)
    ax.plot(eoceno.loc[eoceno.X == 2862].Y, eoceno.loc[eoceno.X == 2862].Z,
             color='m', lw=3.0, label='Eoceno')
    ax.plot(paleoceno.loc[paleoceno.X == 2862].Y, paleoceno.loc[paleoceno.X == 2862].Z,
             color='lime', lw=3.0, label='Paleoceno')

    ax.set_ylabel("Two way time [ms]", fontsize=15)

    ax.axvline(x=5553, color='k')

axs[0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.75, 1.0), facecolor='white', edgecolor='white')
axs[0].set_xticks([])
plt.tight_layout()
plt.savefig('Figures/image_inversion_497_well.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(8, 3))

plt.plot(df.loc[df.WELL=='6-BRSA-497-ESS'].TWT, df.loc[df.WELL=='6-BRSA-497-ESS'].IP_UPS, label='Obs', lw=2.0)
plt.plot(df.loc[df.WELL=='6-BRSA-497-ESS'].TWT, np.exp(mback_ls[:, 5553-xl_start]), label='Back', lw=1.6, ls='-.')
plt.plot(df.loc[df.WELL=='6-BRSA-497-ESS'].TWT, minv_tbt[:, 5553-xl_start], label='Tbt', lw=1.6, ls='--')
#plt.plot(new_twt, minv_hop[:, 1469-new_xl_start], label='Hor', lw=1.6, ls='--')
#plt.plot(new_twt, minv_dop[:, 1469-new_xl_start], label='Dir', lw=1.6, ls='--')
#plt.plot(new_twt, bsd[1269 - new_il_start, 1469-new_xl_start, :], label='Petrel', lw=1.6, ls='--')

plt.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), edgecolor='w', facecolor='w', ncol=5)

plt.xlim(3000, 3600)
plt.ylim(4000, 10000)
plt.savefig('Figures/inversion_497_well.png', dpi=200, bbox_inches='tight')

end_time = time.time()
print("Done!")
print(f"Elapsed time [sec]: {(end_time-start_time):.2f}s")