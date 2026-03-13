import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # Example use of scipy, the function to calculate the jacobian could be manually defined or use a library
from scipy.differentiate import jacobian,hessian
from cyipopt import minimize_ipopt

# System Globals
F_C              = 5220*10**6    # 802.11 Channel 44 Base Frequency
TOTAL_BANDWIDTH = 20*10**6      # 20MHz Channel Bandwidth
SCS             = 312.5*10**3   # 312.5kHz Subcarrier Spacing
N_SC            = 64            # 64 Subcarriers
T_PROP          = 16.7*10**(-9) # 16.7ns Propagation Time
PI              = np.pi

# Hardware Phase Distortion
z   = PI/4          # CFO Phase Offset
sfo = 100*10**-9    # SFO Linear Phase

# Multipath Params
L = 3
multipath_delays = np.array([22, 25, 28, 50])*10**-9
multipath_alphas = np.array([0.25, 0.20, 0.15, 0.10])

def constructMultipath():
    f_sc = np.linspace(F_C, F_C+(N_SC*SCS), N_SC)
    alpha_los = 0.75
    csi_data = alpha_los*np.exp(-1j*2*PI*f_sc*T_PROP)

    dirty_csi_data = csi_data*np.exp(1j*(z+sfo*f_sc))

    # Multipath
    for i in range(L):
        mp_phis = -1j*2*PI*f_sc*multipath_delays[i]
        mp_alphas = 0.25*alpha_los
        multipath_csi = mp_alphas*np.exp(mp_phis)
        dirty_csi_data += multipath_csi
    return csi_data, dirty_csi_data, f_sc

### Channel Estimation ###
# Channel Model =  alpha * exp(j*(beta - 2*pi*fsc*delta)* sum_k(G_k * exp(j * phi_k))
def objective(x, C_measured, f_sc):
    """
    Var      |              Description             |   #   |
    ---------|--------------------------------------|-------|
    alpha_hw |  Hardware Gain                       |   1   |
    alpha_i  |  Gain for ith path                   |   L   |
    phi_i    |  Linear phase for ith path           |   L   |
    delta    |  Linear component of hardware phase  |   1   |
    beta     |  Constant hardware phase component   |   1   |
     
    x array structure (length 2L + 1):
    x[0:L]   = a_i 
    x[L:2L]  = phi_i 
    x[-3]    = a_hw
    x[-2]    = delta
    x[-1]    = beta 
    """
    a_i   = x[0:L]
    phi_i = x[L:2*L] * 1e-9 
    a_hw  = x[-3]
    delta = x[-2]
    beta  = x[-1]
    
    C_est = np.zeros(N_SC, dtype=complex)
    
    for i in range(L):
        C_est += a_i[i] * np.exp(-1j * 2 * PI * f_sc * phi_i[i])
        
    C_est *= a_hw * np.exp(1j * (beta -2*PI*f_sc*delta))
    
    error = C_measured - C_est
    return np.sum(np.abs(error)**2)

def runOptimization(dirty_csi_data, f_sc):
    x0 = np.zeros(2*L + 3)

    # Initial Guesses
    x0[0:L] = 0.5            
    x0[L:2*L] = [10, 30, 50]
    x0[-3] = 1  #CHANGE
    x0[-2] = 100*10**-9    
    x0[-1] = PI/4
    bounds = []
    for i in range(L): bounds.append((0, L))        # Amplitudes
    for i in range(L): bounds.append((0, 500))         # Delays 
    bounds.append((0, 1))                            # HW Gain
    bounds.append((-PI, PI))                           # Delta
    bounds.append((-PI, PI))                           # Beta

    print("Starting delay-domain optimization...")
    res = minimize_ipopt(
        fun=objective, 
        x0=x0, 
        args=(dirty_csi_data, f_sc), 
        bounds=bounds,
        options={'disp': 5, 'max_iter': 500} 
    )

    print(f"Optimization Success: {res.success}")
    print(f"Extracted Amplitudes: {res.x[0:L]}")
    print(f"Extracted Delays (ns): {res.x[L:2*L]}")
    print(f"Extracted Constant Phase: {res.x[-1]}")
    return res

### PLOTTING ###
def plotPerformance(res, csi_data, dirty_csi_data, f_sc):
    indices = np.arange(N_SC)
    optimal_a_i = res.x[0:L]
    phi_i = res.x[L:2*L] * 1e-9 
    optimal_beta = res.x[-1]

    # Reconstruct the Estimated Channel
    C_est = np.zeros(N_SC, dtype=complex)
    for i in range(L):
        C_est += optimal_a_i[i] * np.exp(-1j * 2 * PI * f_sc * phi_i[i])
    C_est *= np.exp(1j * optimal_beta)

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

    fig2, (ax12, ax22) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Magnitude (Gain)
    ax12.plot(indices, np.abs(dirty_csi_data), label='Original Measured CSI', 
            marker='o', linestyle='-', color='blue', alpha=0.7)
    ax12.plot(indices, np.abs(C_est), label='Reconstructed Estimated CSI', 
            marker='x', linestyle='--', color='red')
    ax12.set_title("Channel Magnitude vs Subcarrier")
    ax12.set_xlabel("Subcarrier Index")
    ax12.set_ylabel("Magnitude (Linear)")
    ax12.legend()
    ax12.grid(True)

    # Plot 2: Phase (Unwrapped)
    original_phase = np.unwrap(np.angle(dirty_csi_data))
    estimated_phase = np.unwrap(np.angle(C_est))

    ax12.plot(indices, np.abs(dirty_csi_data), label='Original Measured CSI', marker='o', color='blue', alpha=0.6)

    ax22.plot(indices, original_phase, label='Original Measured Phase', 
            marker='o', linestyle='-', color='blue', alpha=0.7)
    ax22.plot(indices, estimated_phase, label='Reconstructed Estimated Phase', 
            marker='x', linestyle='--', color='red')
    ax22.set_title("Channel Phase vs Subcarrier")
    ax22.set_xlabel("Subcarrier Index")
    ax22.set_ylabel("Unwrapped Phase (Radians)")
    ax22.legend()
    ax22.grid(True)


    plt.tight_layout()
    plt.show()

def main():
    #Do stuff
    csi_data, mp_csi, f_sc = constructMultipath()
    res = runOptimization(mp_csi, f_sc)
    plotPerformance(res, csi_data, mp_csi, f_sc)

if __name__ == "__main__":
    main()
