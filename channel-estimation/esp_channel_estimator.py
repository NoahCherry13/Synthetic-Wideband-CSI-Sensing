import numpy as np
import matplotlib.pyplot as plt
from cyipopt import minimize_ipopt

# --- 1. System Parameters ---
F_C1    = 2422e6  # Center of Ch 1 (HT40 Above)
F_C13   = 2462e6  # Center of Ch 13 (HT40 Below)
SCS     = 312500  # 312.5 kHz Subcarrier Spacing
L       = 8       # Number of paths to estimate

def get_phys_freqs(center_freq, n_subcarriers=114):
    """
    Maps indices to physical frequencies. 
    Assumes the 114 subcarriers are the 'live' ones from a shifted 128-FFT.
    """
    indices = np.arange(-n_subcarriers//2, n_subcarriers//2)
    return center_freq + (indices * SCS)

def find_global_initial_guess(C_obs, F_obs, L):
    """
    Performs a brute-force scan to find the strongest LoS path.
    This prevents the optimizer from getting stuck in 4e5 residual local minima.
    """
    scan_delays = np.linspace(0, 150, 300)
    correlations = []
    
    for t in scan_delays:
        test_model = np.exp(-1j * 2 * np.pi * F_obs * t * 1e-9)
        correlations.append(np.abs(np.vdot(test_model, C_obs)))
    
    best_tau = scan_delays[np.argmax(correlations)]
    
    # 2. Assemble x0: [a1..aL, phi1..phiL, gamma, delta, beta1, beta_rel]
    x0 = np.zeros(2*L + 4)
    x0[0] = 1.0                      # Initial Gain (normalized)
    x0[L] = best_tau                 # Initial Delay from Scan
    
    # Spread Delays in time to prevent convergence to same values
    for i in range(1, L):
        x0[i] = 0.5 / (i + 1)               
        x0[L+i] = best_tau + (i * 12.5)
        
    x0[-4] = 1.0                     # gamma_rel
    x0[-3] = 0.0                     # delta_rel (ns)
    x0[-2] = np.angle(C_obs[0])      # beta1
    x0[-1] = 0.0                     # beta_rel
    
    return x0


def objective(x, C_meas, F1, F13, L):
    a_i = x[0:L]
    phi_i = x[L:2*L] * 1e-9
    gamma_rel, delta_rel, beta1, beta_rel = x[-4:]
    delta_rel *= 1e-9 # Convert ns to seconds

    # Calculate Channel 1 Estimate
    C_est1 = np.zeros(len(F1), dtype=complex)
    for k in range(L):
        C_est1 += a_i[k] * np.exp(-1j * 2 * np.pi * F1 * phi_i[k])
    C_est1 *= np.exp(1j * beta1)

    # Calculate Channel 13 Estimate(with hardware corrections)
    C_est2 = np.zeros(len(F13), dtype=complex)
    for k in range(L):
        C_est2 += a_i[k] * np.exp(-1j * 2 * np.pi * F13 * phi_i[k])

    # Reconstruct Channel 2
    C_est2 *= gamma_rel * np.exp(1j * (beta1 + beta_rel - 2 * np.pi * F13 * delta_rel))

    C_model = np.concatenate((C_est1, C_est2))
    
    # MMSE Residual
    return np.sum(np.abs(C_meas - C_model)**2)



def plot_results(res, C_obs, F1, F13, L, norm_factor):
    """
    Visualizes the optimization fit and environment multipath.
    Calculates RMSE and Phase Error for performance tracking.
    """
    # 1. Extract and Unpack Optimized Parameters
    a_opt = res.x[0:L]
    phi_opt = res.x[L:2*L] # Delays in ns
    gamma_rel, delta_rel, beta1, beta_rel = res.x[-4:]
    delta_sec = delta_rel * 1e-9

    # 2. Reconstruct the Model for Channel 1 and Channel 13
    C_model1 = np.zeros(len(F1), dtype=complex)
    for k in range(L):
        C_model1 += a_opt[k] * np.exp(-1j * 2 * np.pi * F1 * phi_opt[k] * 1e-9)
    C_model1 *= np.exp(1j * beta1)

    C_model13 = np.zeros(len(F13), dtype=complex)
    for k in range(L):
        C_model13 += a_opt[k] * np.exp(-1j * 2 * np.pi * F13 * phi_opt[k] * 1e-9)
    C_model13 *= gamma_rel * np.exp(1j * (beta1 + beta_rel - 2 * np.pi * F13 * delta_sec))

    # Combine for plotting
    C_model = np.concatenate((C_model1, C_model13))
    F_total = np.concatenate((F1, F13)) / 1e9 # GHz

    # 3. Calculate Performance Metrics
    rmse = np.sqrt(np.mean(np.abs(C_obs - C_model)**2))
    # Mean Absolute Phase Error (MAPE) - skip the gap for better stats
    phase_error = np.mean(np.abs(np.angle(C_obs) - np.angle(C_model)))

    # 4. Create the Figure
    plt.style.use('seaborn-v0_8-paper') 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), constrained_layout=True)
    fig.suptitle(f'Stitched 80MHz CSI Analysis (L={L} Paths)', fontsize=16, fontweight='bold')

    # --- Subplot 1: Magnitude Fit ---
    ax1.plot(F_total, np.abs(C_obs), 'ko', markersize=3, alpha=0.3, label='Measured (Normalized)')
    ax1.plot(F_total, np.abs(C_model), 'r-', linewidth=2, label='Optimizer Fit')
    ax1.axvline(x=F1[-1]/1e9, color='green', linestyle='--', alpha=0.5, label='Channel Boundary')
    ax1.set_ylabel('Normalized Magnitude', fontweight='bold')
    ax1.set_title('Spectral Magnitude: Frequency Selective Fading')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 2: Unwrapped Phase (The "Delay Slope") ---
    # We unwrap the phase to show the continuous slope across the gap
    obs_phase = np.unwrap(np.angle(C_obs))
    model_phase = np.angle(C_model)
    
    ax2.plot(F_total, obs_phase, 'ko', markersize=3, alpha=0.3)
    ax2.plot(F_total, model_phase, 'b-', linewidth=2, label='Phase Model')
    ax2.set_ylabel('Phase (Radians)', fontweight='bold')
    ax2.set_title('Unwrapped Phase: Delay Slope Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 3: Power Delay Profile (PDP) ---
    # Show where the optimizer placed the discrete paths compared to the IFFT
    n_fft = 1024
    pdp_raw = np.abs(np.fft.ifft(C_obs, n=n_fft))
    # Approximate time axis based on 80MHz span
    t_axis = np.linspace(0, (1/80e6)*n_fft, n_fft) * 1e9 # ns

    ax3.plot(t_axis[:200], pdp_raw[:200], color='gray', alpha=0.5, label='Measured PDP (IFFT)')
    for k in range(L):
        ax3.vlines(phi_opt[k], 0, a_opt[k], colors=f'C{k}', linestyles='solid', 
                   linewidth=3, label=f'Path {k+1}: {phi_opt[k]:.2f} ns')
    
    ax3.set_xlabel('Delay (ns)', fontweight='bold')
    ax3.set_ylabel('Path Weight (a_i)', fontweight='bold')
    ax3.set_title('Time Domain: Multipath Resolution')
    ax3.set_xlim(0, 150)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    stats_text = (
        f"--- Performance ---\n"
        f"Status: {'SUCCESS' if res.success else 'FAILED'}\n"
        f"Final Residual: {res.fun:.2e}\n"
        f"RMSE: {rmse:.4f}\n"
        f"Phase Error: {phase_error:.3f} rad\n\n"
        f"--- HW Params ---\n"
        f"Beta Rel: {beta_rel:.3f} rad\n"
        f"Delta Rel: {delta_rel:.2f} ns"
    )
    fig.text(1.02, 0.5, stats_text, fontsize=11, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), va='center')

    plt.show()

def main():

    n_avg = 10
    try:
        data_ch1 = np.load("csi_processed_ch1.npy") 
        data_ch13 = np.load("csi_processed_ch13.npy")
    except FileNotFoundError:
        print("Error: .npy files not found. Ensure they are in the same folder.")
        return

    packet_idx = 0
    c1 = data_ch1[9]
    c13 = data_ch13[31]
    
    C_obs = np.concatenate((c1, c13))
    F1 = get_phys_freqs(F_C1, len(c1))
    F13 = get_phys_freqs(F_C13, len(c13))
    F_obs = np.concatenate((F1, F13))

    norm_factor = np.mean(np.abs(C_obs))
    C_norm = C_obs / norm_factor

    x0 = find_global_initial_guess(C_norm, F_obs, L)

    bounds = ([(0, 2)] * L + 
              [(0, 150)] * L + 
              [(0.5, 2.0), (-20, 20), (-np.pi, np.pi), (-np.pi, np.pi)])

    print("Starting IPOPT Optimization...")
    res = minimize_ipopt(objective, x0=x0, args=(C_norm, F1, F13, L), 
                         bounds=bounds, options={b'max_iter': 1000, b'tol': 1e-7})

    print(f"Optimization Success: {res.success}")
    print(f"Final Residual: {res.fun:.4e}")
    print(f"Estimated Delays: {res.x[L:2*L]} ns")

    # Final Plot
    plot_results(res, C_norm, F1, F13, L, norm_factor)

if __name__ == "__main__":
    main()