import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # Example use of scipy, the function to calculate the jacobian could be manually defined or use a library
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
    a_i    = x[0:L]
    phi_i  = x[L:2*L] * 1e-9 
    a_hw1  = x[-6]
    a_hw2  = x[-5]
    delta1 = x[-4]
    delta2 = x[-3]
    beta1  = x[-2]
    beta2  = x[-1]
    
    C_est1 = np.zeros(N_SC, dtype=complex)
    C_est2 = np.zeros(N_SC, dtype=complex)

    for i in range(L):
        C_est1 += a_i[i] * np.exp(-1j * 2 * PI * F_SC1 * phi_i[i]) 
        C_est2 += a_i[i] * np.exp(-1j * 2 * PI * F_SC2 * phi_i[i])
    
    C_est1 *= a_hw1 * np.exp(1j * (beta1 -2*PI*F_SC1*delta1))
    C_est2 *= a_hw2 * np.exp(1j * (beta2 -2*PI*F_SC2*delta2))
    C_composite = np.concatenate((C_est1, C_est2))

    error = C_measured - (C_composite)
    print(np.sum(np.abs(error)**2))
    return np.sum(np.abs(error)**2)

def runOptimization(c_ref, L, x0):

    bounds = []
    for i in range(L): bounds.append((0, L))        # Amplitudes
    for i in range(L): bounds.append((0, 200))         # Delays 
    bounds.append((0, 1))                            # HW Gain
    bounds.append((0, 1))                            # HW Gain
    bounds.append((-PI, PI))                           # Delta
    bounds.append((-PI, PI))                           # Delta
    bounds.append((-PI, PI))                           # Beta
    bounds.append((-PI, PI))                           # Beta


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
    print(f"Extracted Delays (ns): {res.x[L:2*L]}")
    print(f"Extracted Constant Phase: {res.x[-1]}")
    print(f"Objective Value: {res.fun}")
    return res

### PLOTTING ###
def plotPerformance(channel1, channel2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Amplitude
    ax1.plot(np.arange(len(channel1)), np.abs(channel1), marker='o', color='blue')
    ax1.plot(np.arange(len(channel2)), np.abs(channel2), marker='o', color='red')
    ax1.set_title("Gain/Frequency")
    ax1.set_xlabel("Subcarrier")
    ax1.set_ylabel("Gain")
    ax1.grid(True)

    # Plot 2: Phase
    ax2.plot(np.arange(len(channel1)), np.unwrap(np.angle(channel1)),  marker='o', color='blue')
    ax2.plot(np.arange(len(channel2)), np.unwrap(np.angle(channel2)),  marker='x', color='red')
    ax2.set_title("Phase/Frequency")
    ax2.set_xlabel("Subcarrier")
    ax2.set_ylabel("Phase")
    ax2.grid(True)

    # fig2, (ax12, ax22) = plt.subplots(1, 2, figsize=(14, 5))



    plt.tight_layout()
    plt.show()

def main():
    #Do stuff
    # Hardware Phase Distortion
    z   = [PI/4, PI/6]          # CFO Phase Offset
    sfo = [100*10**-9, 80*10**-9 ]    # SFO Linear Phase

    # Multipath Params
    L = 3
    multipath_delays = np.array([16.7, 22, 25, 28, 50])*1e-9
    multipath_alphas = np.array([0.75, 0.25, 0.20, 0.15, 0.10])
    channel1 = constructMultipath(F_SC1, multipath_delays, multipath_alphas, sfo[0], z[0], L)
    channel2 = constructMultipath(F_SC2, multipath_delays, multipath_alphas, sfo[1], z[1], L)
    
    # Parameter Estimation
    x0 = np.zeros(2*L + 6)
    # Initial Guesses
    x0[0:L] = [0.75, 0.25, 0.20]            
    x0[L:2*L] = [16.7, 22, 25]
    x0[-6] = 1
    x0[-5] = 1
    x0[-4] = 100*10**-9
    x0[-3] = 80*10**-9
    x0[-2] = PI/4    
    x0[-1] = PI/6

    C_ref = np.concatenate((channel1, channel2))
    res = runOptimization(C_ref, L, x0)

    '''
     csi_data = los_gain*np.exp(-1j*2*PI*frequencies*T_PROP)
    dirty_csi_data = csi_data*np.exp(1j*(z+sfo*frequencies))

    # Multipath
    for i in range(L-1):
        mp_phis = -1j*2*PI*frequencies*delays[i]
        mp_alpha = gains[i]
        multipath_csi = mp_alpha*np.exp(mp_phis)
        dirty_csi_data += multipath_csi
    return csi_data, dirty_csi_data
    '''
    #plotting
    a_i    = x0[0:L]
    phi_i  = x0[L:2*L] * 1e-9 
    a_hw1  = x0[-6]
    a_hw2  = x0[-5]
    delta1 = x0[-4]
    delta2 = x0[-3]
    beta1  = x0[-2]
    beta2  = x0[-1]
    
    C_est1 = np.zeros(N_SC, dtype=complex)
    C_est2 = np.zeros(N_SC, dtype=complex)

    for i in range(L):
        C_est1 += a_i[i] * np.exp(-1j * 2 * PI * F_SC1 * phi_i[i]) 
        C_est2 += a_i[i] * np.exp(-1j * 2 * PI * F_SC2 * phi_i[i])
    
    C_est1 *= a_hw1 * np.exp(1j * (beta1 -2*PI*F_SC1*delta1))
    C_est2 *= a_hw2 * np.exp(1j * (beta2 -2*PI*F_SC2*delta2))
    C_composite = np.concatenate((C_est1, C_est2))
    
    plotPerformance(C_ref, C_composite)

if __name__ == "__main__":
    main()
