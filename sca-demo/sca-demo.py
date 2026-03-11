import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # Example use of scipy, the function to calculate the jacobian could be manually defined or use a library
from scipy.differentiate import jacobian,hessian

# System Globals
F_C              = 5220*10**6    # 802.11 Channel 44 Base Frequency
TOTAL_BANDWIDTH = 20*10**6      # 20MHz Channel Bandwidth
SCS             = 312.5*10**3   # 312.5kHz Subcarrier Spacing
N_SC            = 64            # 64 Subcarriers
T_PROP          = 16.7*10**(-9) # 16.7ns Propagation Time
PI              = np.pi

def calculatePhis(t_prop):
    return -1j*2*PI*f_sc*t_prop

def calculateCSI(alphas, phis):
    return alphas*np.exp(phis)
# Channel Parameters
f_sc = np.linspace(F_C, F_C+(N_SC*SCS), N_SC)
alphas = np.ones(N_SC)
csi_data = alphas*np.exp(-1j*2*PI*f_sc*T_PROP)

# Hardware Phase Distortion
z   = PI/4          # CFO Phase Offset
sfo = 100*10**-9    # SFO Linear Phase

dirty_csi_data = csi_data*np.exp(1j*(z+sfo*f_sc))

# Multipath
n_mp = 4
multipath_delays = np.array([22, 25, 28, 50])*10**-9
multipath_alphas = np.array([0.25, 0.20, 0.15, 0.10])
for i in range(n_mp):
    mp_phis = calculatePhis(multipath_delays[i])
    mp_alphas = 0.25*alphas
    multipath_csi = calculateCSI(mp_alphas, mp_phis)
    dirty_csi_data += multipath_csi

### Channel Estimation ###
# Channel Model =  alpha * sum_k(G_k * exp(j * phi_k)) * exp(j * psi)

### PLOTTING ###
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Amplitude
ax1.plot(np.arange(N_SC), np.abs(csi_data), marker='o', color='blue')
ax1.plot(np.arange(N_SC), np.abs(dirty_csi_data), marker='o', color='blue')
ax1.set_title("Gain/Frequency")
ax1.set_xlabel("Subcarrier")
ax1.set_ylabel("Gain")
ax1.grid(True)

# Plot 2: Phase
ax2.plot(np.arange(N_SC), np.unwrap(np.angle(csi_data)),  marker='o', color='blue')
ax2.plot(np.arange(N_SC), np.unwrap(np.angle(dirty_csi_data)),  marker='x', color='red')
ax2.set_title("Phase/Frequency")
ax2.set_xlabel("Subcarrier")
ax2.set_ylabel("Phase")
ax2.grid(True)


plt.tight_layout()
plt.show()