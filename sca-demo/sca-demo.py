import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from scipy.differentiate import jacobian,hessian
from cyipopt import minimize_ipopt

# System Globals
F_C1            = 5220*10**6    # 802.11 Channel 44 Base Frequency
F_C2            = 5240*10**6    # 802.11 Channel 48 Base Frequency
TOTAL_BANDWIDTH = 20*10**6      # 20MHz Channel Bandwidth
SCS             = 312.5*10**3   # 312.5kHz Subcarrier Spacing
N_SC            = 64            # 64 Subcarriers
T_PROP          = 16.7*10**(-9) # 16.7ns Propagation Time
PI              = np.pi
F_SC1 = np.linspace(F_C1, F_C1+(N_SC*SCS), N_SC)
F_SC2 = np.linspace(F_C2, F_C2+(N_SC*SCS), N_SC)

def constructMultipath(frequencies, delays, gains, delta, beta, L):
    delta *= 1e-9
    channel = np.zeros(N_SC, dtype=complex)
    for i in range(L):
        channel += gains[i]*np.exp(-1j*2*PI*frequencies*delays[i])

    channel *= 1 * np.exp(1j * (beta -2*PI*frequencies*delta))
    return channel

### Channel Estimation ###
# Channel Model =  alpha * exp(j*(beta - 2*pi*fsc*delta)* sum_k(G_k * exp(j * phi_k))
def objective(x, C_measured, L):
    """
    Var      |              Description             |   #   | Channel |
    ---------|--------------------------------------|-------|---------|
    alpha_hw |  Hardware Gain                       |   1   |    1    |
    alpha_hw |  Hardware Gain                       |   1   |    2    |
    alpha_i  |  Gain for ith path                   |   L   |    x    |
    phi_i    |  Linear phase for ith path           |   L   |    x    |
    delta    |  Linear component of hardware phase  |   1   |    1    |
    delta    |  Linear component of hardware phase  |   1   |    2    |
    beta     |  Constant hardware phase component   |   1   |    1    |
    beta     |  Constant hardware phase component   |   1   |    2    |
     
    x array structure (length 2L + 1):
    x[0:L]   = a_i 
    x[L:2L]  = phi_i 
    x[-3]    = a_hw
    x[-2]    = delta
    x[-1]    = beta 
    """
    a_i    = x[0:L] * 1e-2
    phi_i  = x[L:2*L] * 1e-9 
    delta_rel = x[-3] * 1e-9
    beta1 = x[-2]
    beta_rel =  x[-1]
    
    C_est1 = np.zeros(N_SC, dtype=complex)
    C_est2 = np.zeros(N_SC, dtype=complex)

    for i in range(L):
        C_est1 += a_i[i] * np.exp(-1j * 2 * PI * F_SC1 * phi_i[i]) 
        C_est2 += a_i[i] * np.exp(-1j * 2 * PI * F_SC2 * phi_i[i])
    
    C_est1 *= np.exp(1j*beta1)
    C_est2 *=  np.exp(1j * (beta_rel + beta1 -2*PI*F_SC2*delta_rel))
    C_composite = np.concatenate((C_est1, C_est2))

    error = C_measured - (C_composite)
    return np.sum(np.abs(error)**2)

def runOptimization(c_ref, L, x0):

    bounds = []
    for i in range(L): bounds.append((0, L*1e2))        # Amplitudes
    for i in range(L): bounds.append((0, 150))      # Delays 
    bounds.append((-100, 200))                      # Relative Linear Phase
    bounds.append((-PI/2, PI/2))                    # Baseline Phase Offset
    bounds.append((-PI/2, PI/2))                    # Relative Phase Offset



    print("Starting delay-domain optimization...")
    res = minimize_ipopt(
        fun=objective, 
        x0=x0, 
        args=(c_ref, L), 
        bounds=bounds,
        options={'disp': 0, 'max_iter': 1000} 
    )

    print(f"Optimization Success: {res.success}")
    print(f"Extracted Amplitudes: {res.x[0:L]}")
    print(f"Extracted Apparent Delays (ns): {res.x[L:2*L]}")
    print(f"Extracted Rel Delta (ns): {res.x[-2]}")
    print(f"Extracted Rel Beta (rad): {res.x[-1]}")
    print(f"Objective Value: {res.fun}")
    return res

### PLOTTING ###
def plotPerformance(channel1, channel2, channel_est):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Amplitude
    ax1.plot(np.arange(len(channel1)), np.abs(channel1), marker='o', color='blue')
    #ax1.plot(np.arange(len(channel2)), np.abs(channel2), marker='x', color='red')
    ax1.plot(np.arange(len(channel2)), np.abs(channel_est), marker='s', color='green')

    ax1.set_title("Gain/Frequency")
    ax1.set_xlabel("Subcarrier")
    ax1.set_ylabel("Gain")
    ax1.grid(True)

    # Plot 2: Phase
    ax2.plot(np.arange(len(channel1)), np.unwrap(np.angle(channel1)),  marker='o', color='blue')
    #ax2.plot(np.arange(len(channel2)), np.unwrap(np.angle(channel2)),  marker='x', color='red')
    ax2.plot(np.arange(len(channel2)), np.unwrap(np.angle(channel_est)),  marker='s', color='green')

    ax2.set_title("Phase/Frequency")
    ax2.set_xlabel("Subcarrier")
    ax2.set_ylabel("Phase")
    ax2.grid(True)

    fig2, (ax12, ax22) = plt.subplots(1, 2, figsize=(14, 5))
    ax12.plot(np.arange(len(channel2)), np.abs(channel2), marker='x', color='red')
    ax22.plot(np.arange(len(channel2)), np.unwrap(np.angle(channel2)),  marker='x', color='red')
    ax12.set_title("Gain/Frequency")
    ax12.set_xlabel("Subcarrier")
    ax12.set_ylabel("Gain")
    ax22.grid(True)
    ax22.set_title("Phase/Frequency")
    ax22.set_xlabel("Subcarrier")
    ax22.set_ylabel("Phase")
    ax22.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    #Do stuff
    # Hardware Phase Distortion
    z   = [PI/4, PI/6]          # CFO Phase Offset
    sfo = [100, 80]    # SFO Linear Phase

    # Multipath Params
    L = 3
    multipath_delays = np.array([16.7, 22, 25, 28, 50])*1e-9
    multipath_alphas = np.array([0.75, 0.25, 0.20, 0.15, 0.10])
    channel1 = constructMultipath(F_SC1, multipath_delays, multipath_alphas, sfo[0], z[0], L)
    channel2 = constructMultipath(F_SC2, multipath_delays, multipath_alphas, sfo[1], z[1], L)
    
    # Parameter Estimation
    x0 = np.zeros(2*L + 3)
    # Initial Guesses
    x0[0:L] = [50, 50, 10]           
    x0[L:2*L] = [17.74 + sfo[0], 22.96 + sfo[0], 26.05 + sfo[0]]
    x0[-3] = (sfo[1] - sfo[0]) + 0.04
    x0[-2] = 0.0
    x0[-1] = 0.0
    ### TRUE VALUES FOR TESTING ###
    # x0[0:L] = [0.75, 0.25, 0.20]            
    # x0[L:2*L] = [16.7 + sfo[0], 22 + sfo[0], 25 + sfo[0]]
    # x0[-3] = sfo[1]-sfo[0]
    # x0[-2] = PI/4
    # x0[-1] = z[1]-z[0]

    C_ref = np.concatenate((channel1, channel2))
    res = runOptimization(C_ref, L, x0)

    # Plotting
    a_i    = x0[0:L]
    phi_i  = x0[L:2*L] * 1e-9 
    delta_rel = x0[-3] * 1e-9
    beta1 = x0[-2]
    beta_rel =  x0[-1]
    
    C_est1 = np.zeros(N_SC, dtype=complex)
    C_est2 = np.zeros(N_SC, dtype=complex)

    for i in range(L):
        C_est1 += a_i[i] * 1e-2 * np.exp(-1j * 2 * PI * F_SC1 * phi_i[i]) 
        C_est2 += a_i[i] * 1e-2 * np.exp(-1j * 2 * PI * F_SC2 * phi_i[i])
    
    C_est1 *= np.exp(1j * beta1)
    C_est2 *= np.exp(1j * (beta_rel + beta1 -2*PI*F_SC2*delta_rel))
    C_composite = np.concatenate((C_est1, C_est2))
    
    a_i         = res.x[0:L] * 1e-2
    phi_i       = res.x[L:2*L] * 1e-9 
    delta_rel   = res.x[-3] * 1e-9
    beta1       = res.x[-2]
    beta_rel    = res.x[-1]
    
    C_est1 = np.zeros(N_SC, dtype=complex)
    C_est2 = np.zeros(N_SC, dtype=complex)

    for i in range(L):
        C_est1 += a_i[i] * np.exp(-1j * 2 * PI * F_SC1 * phi_i[i]) 
        C_est2 += a_i[i] * np.exp(-1j * 2 * PI * F_SC2 * phi_i[i])
    
    C_est1 *= np.exp(1j * beta1)
    C_est2 *= np.exp(1j * (beta_rel + beta1 -2*PI*F_SC2*delta_rel))
    C_est_opt = np.concatenate((C_est1, C_est2))

    correction_factor = np.exp(-1j * (beta_rel - 2 * PI * F_SC2 * delta_rel))
    channel2_aligned = C_est2 * correction_factor
    
    # Combine the frequency axes and the aligned data
    C_stitched_measured = np.concatenate((channel1, channel2_aligned))

    plotPerformance(C_ref, C_stitched_measured, C_est_opt)

if __name__ == "__main__":
    main()
