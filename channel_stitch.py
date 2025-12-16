import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.widgets import Slider, Button, TextBox

def simulate_ideal_channel(num_subcarriers_total, subcarrier_spacing_hz, multipath_components):
    """
    Simulates the ideal, coherent wideband channel frequency response.

    Args:
        num_subcarriers_total (int): Total number of subcarriers (e.g., 256 for 80MHz).
        subcarrier_spacing_hz (float): Frequency spacing between subcarriers (e.g., 312.5e3).
        multipath_components (list): A list of tuples, where each tuple is (complex_gain, delay_sec).
                                     e.g., [(1.0, 100e-9), (0.5*np.exp(1j*np.pi/2), 300e-9)]

    Returns:
        np.ndarray: A 1D complex array of the ideal channel frequency response (C_ideal).
    """
    # Create an array of subcarrier frequencies
    frequencies_hz = np.arange(num_subcarriers_total) * subcarrier_spacing_hz
    
    C_ideal = np.zeros(num_subcarriers_total, dtype=complex)
    
    # Sum the contributions from each multipath component
    # C_k = sum_l(g_l * exp(-j * 2 * pi * f_k * tau_l))
    if not multipath_components: # Handle case of 0 paths
        return C_ideal + 1e-9 # Return small noise floor
        
    for gain, delay in multipath_components:
        C_ideal += gain * np.exp(-1j * 2 * np.pi * frequencies_hz * delay)
        
    return C_ideal

def apply_channel_distortions(C_ideal, num_channels, subcarriers_per_channel, distortion_params):
    """
    Applies independent distortions to each channel segment of the ideal channel.

    Args:
        C_ideal (np.ndarray): The ideal wideband channel response.
        num_channels (int): Number of channels to split into (e.g., 4).
        subcarriers_per_channel (int): Subcarriers per channel (e.g., 64).
        distortion_params (list): List of tuples, one per channel.
                                  Each tuple is (alpha_complex, phi_slope).
                                  alpha_complex: Complex gain (amplitude + phase offset).
                                  phi_slope: Linear phase slope (phase change per subcarrier).

    Returns:
        np.ndarray: A 1D complex array of the distorted, stitched channel.
    """
    C_measured_stitched = np.zeros_like(C_ideal, dtype=complex)
    
    # Create the local subcarrier index array (e.g., [0, 1, ..., 63])
    k_local = np.arange(subcarriers_per_channel)
    
    for i in range(num_channels):
        # Get parameters for this channel
        alpha_i, phi_i = distortion_params[i]
        
        # Calculate the global indices for this channel
        start_idx = i * subcarriers_per_channel
        end_idx = (i + 1) * subcarriers_per_channel
        
        # Get the corresponding slice from the ideal channel
        ideal_slice = C_ideal[start_idx:end_idx]
        
        # Create the distortion array for this channel
        # D_i[k] = alpha_i * exp(-j * phi_i * k_local)
        distortion_array = alpha_i * np.exp(-1j * phi_i * k_local)
        
        # Apply the distortion
        C_measured_stitched[start_idx:end_idx] = ideal_slice * distortion_array
        
    return C_measured_stitched

def correct_iterative_approach(C_measured, num_channels, subcarriers_per_channel):
    """
    Implements the iterative correction approach from the slides.
    Assumes Ch 0 is the reference. Corrects Ch 1 based on Ch 0,
    Ch 2 based on *corrected* Ch 1, and so on.
    
    Args:
        C_measured (np.ndarray): The distorted, stitched channel (now with noise).
        num_channels (int): Number of channels (e.g., 4).
        subcarriers_per_channel (int): Subcarriers per channel (e.g., 64).

    Returns:
        tuple: (C_corrected, global_gammas)
            C_corrected (np.ndarray): The iteratively corrected channel.
            global_gammas (list): List of the final complex correction
                                  factor (Γ) applied to each channel.
    """
    C_corrected = np.copy(C_measured)
    
    # global_gammas[i] is the total correction factor applied to channel i
    # This is the 'α(0)' from your "Global Approach" slide
    global_gammas = [1.0 + 0j] * num_channels # Ch 0 is reference
    
    # cumulative_gamma tracks the compounded error
    cumulative_gamma = 1.0 + 0j
    
    # Get the reference slice (Channel 0)
    C_ref_slice = C_corrected[0 : subcarriers_per_channel]

    # Loop from the second channel (i=1) onwards
    for i in range(1, num_channels):
        
        # C1 is the last subcarrier of the *previous corrected* channel
        C1 = C_ref_slice[-1] 
        
        # Get the *original measured* data for the current channel (i)
        idx_i_start = i * subcarriers_per_channel
        idx_i_end = (i + 1) * subcarriers_per_channel
        C_i_measured_slice = C_measured[idx_i_start:idx_i_end]
        
        # C2 is the first subcarrier of the *current measured* channel
        C2 = C_i_measured_slice[0]
        
        # Calculate the local correction factor based on the boundary
        # Γ_local = C2 / C1
        # Add small epsilon to prevent division by zero if C1 is in a deep fade
        local_gamma = C2 / (C1 + 1e-12)
        
        # The global correction for this channel is compounded
        # Γ_global_i = Γ_local_i * Γ_global_i-1
        cumulative_gamma = cumulative_gamma * local_gamma
        global_gammas[i] = cumulative_gamma
        
        # Apply the global correction to the measured data for this channel
        C_corrected_slice = C_i_measured_slice / cumulative_gamma
        
        # Store the corrected data
        C_corrected[idx_i_start:idx_i_end] = C_corrected_slice
        
        # This corrected slice becomes the reference for the next iteration
        C_ref_slice = C_corrected_slice
        
    return C_corrected, global_gammas

def estimate_slopes_from_phase_diff(phase_difference, num_channels, subcarriers_per_channel):
    """
    Estimates the linear phase slope (φ_i) for each channel
    by fitting a line to the *phase difference* between the
    measured and iteratively corrected channels.
    
    Args:
        phase_difference (np.ndarray): Unwrapped phase(C_measured) - Unwrapped phase(C_iterative).
        num_channels (int): Number of channels (e.g., 4).
        subcarriers_per_channel (int): Subcarriers per channel (e.g., 64).

    Returns:
        list: A list of the estimated slopes (phi_i) in radians per subcarrier.
    """
    slopes = []
    local_indices = np.arange(subcarriers_per_channel)
    
    for i in range(num_channels):
        # Get the phase difference slice for this channel
        start_idx = i * subcarriers_per_channel
        end_idx = (i + 1) * subcarriers_per_channel
        phase_diff_slice = phase_difference[start_idx:end_idx]
        
        # Fit a 1st-degree polynomial (a line) to the phase difference
        # The true phase difference is (-phi_i * k_local), so the slope is -phi_i.
        slope, intercept = np.polyfit(local_indices, phase_diff_slice, 1)
        
        # We return -slope because phase_diff = angle(ideal) - angle(dist) - angle(ideal)
        # angle(dist) = -phi_i * k
        # So, phase_diff = -(-phi_i * k) = phi_i * k. Slope is phi_i.
        # Let's re-check:
        # angle(C_meas) = angle(C_ideal) + angle(Dist)
        # angle(C_iter) ~ angle(C_ideal)
        # diff = angle(C_meas) - angle(C_iter) ~ angle(Dist) = -phi_i * k
        # So the slope of the diff is -phi_i. We want phi_i.
        slopes.append(-slope)
        
    return slopes


def initialize_global_approach(C_iterative_corrected, subcarrier_spacing_hz, num_peaks):
    """
    Performs the IFFT on the iteratively corrected channel to find
    the largest multipath components (g(0)) and their delays (τ(0)).
    
    Args:
        C_iterative_corrected (np.ndarray): The channel from correct_iterative_approach.
        subcarrier_spacing_hz (float): e.g., 312.5e3
        num_peaks (int): The number of paths (T) to find.

    Returns:
        tuple: (g_0, tau_0)
            g_0 (np.ndarray): Array of complex gains for the T largest peaks.
            tau_0 (np.ndarray): Array of delays (in seconds) for the T largest peaks.
    """
    print(f"\nInitializing Global Approach: Finding {num_peaks} largest paths...")
    
    # Compute the Channel Impulse Response (CIR) via IFFT
    # We use fftshift to center the 0-delay component
    cir = np.fft.fftshift(np.fft.ifft(C_iterative_corrected))
    
    num_subcarriers = len(C_iterative_corrected)
    total_bandwidth = num_subcarriers * subcarrier_spacing_hz
    
    # Time resolution of the CIR
    time_resolution = 1.0 / total_bandwidth
    
    # Create the time (delay) axis for the CIR
    # Centered around 0
    delays_sec = np.arange(-num_subcarriers/2, num_subcarriers/2) * time_resolution
    
    # Find the indices of the 'num_peaks' largest magnitudes
    magnitudes = np.abs(cir)
    peak_indices = np.argsort(magnitudes)[-num_peaks:][::-1] # Largest to smallest
    
    # Get the complex gain and delay for these peaks
    g_0 = cir[peak_indices]
    tau_0 = delays_sec[peak_indices]
    
    return g_0, tau_0

def pack_parameters(g, tau, alphas, phis):
    """
    Packs all optimization parameters into a 1D real array for scipy.optimize.
    We fix alpha[0] = 1 and phi[0] = 0, so we don't optimize them.
    
    *** Parameters are SCALED here to be on a similar order of magnitude ***
    """
    # g is complex (T vals) -> 2*T real (Order ~1.0)
    g_real = np.real(g)
    g_imag = np.imag(g)
    
    # tau is real (T vals) -> T real (Order ~1e-7)
    # Scale from seconds to nanoseconds (Order ~100.0)
    tau_scaled = tau * 1e9
    
    # alphas is complex (I vals) -> 2*(I-1) real (skip first) (Order ~1.0)
    alphas_real = np.real(alphas[1:])
    alphas_imag = np.imag(alphas[1:])
    
    # phis is real (I vals) -> (I-1) real (skip first) (Order ~0.01)
    # Scale by 100 (Order ~1.0)
    phis_scaled = phis[1:] * 100
    
    return np.concatenate([g_real, g_imag, tau_scaled, alphas_real, alphas_imag, phis_scaled])

def unpack_parameters(x, num_peaks, num_channels):
    """
    Unpacks the 1D real array from scipy.optimize back into named parameters.
    
    *** Parameters are UN-SCALED here back to their original units ***
    """
    T = num_peaks
    I = num_channels
    
    # Unpack g (complex)
    g_real = x[0:T]
    g_imag = x[T:2*T]
    g = g_real + 1j * g_imag
    
    # Unpack tau (scaled)
    tau_scaled = x[2*T : 3*T]
    tau = tau_scaled / 1e9 # Convert nanoseconds back to seconds
    
    # Unpack alphas (complex)
    alphas_real = x[3*T : 3*T + (I-1)]
    alphas_imag = x[3*T + (I-1) : 3*T + 2*(I-1)]
    # Add back the fixed alpha[0] = 1
    alphas = np.concatenate([ [1.0 + 0j], alphas_real + 1j * alphas_imag ])
    
    # Unpack phis (scaled)
    phis_scaled = x[3*T + 2*(I-1) : 3*T + 3*(I-1)]
    phis_real = phis_scaled / 100 # Scale slopes back down
    # Add back the fixed phi[0] = 0
    phis = np.concatenate([ [0.0], phis_real ])
    
    return g, tau, alphas, phis

def global_objective_function(x, C_measured, N_CH, N_SUB, N_PEAKS, SUB_SPACING_HZ):
    """
    The objective function for the global minimization.
    Calculates the sum of squared errors between the measured CSI
    and the channel model built from the parameters in x.
    
    Args:
        x (np.ndarray): 1D array of packed parameters (g, tau, alpha, phi).
        C_measured (np.ndarray): The original distorted, stitched channel (now with noise).
        N_CH (int): Number of channels.
        N_SUB (int): Subcarriers per channel.
        N_PEAKS (int): Number of multipath components (T).
        SUB_SPACING_HZ (float): Frequency spacing.

    Returns:
        float: The sum of squared errors.
    """
    
    # 1. Unpack parameters
    g, tau, alphas, phis = unpack_parameters(x, N_PEAKS, N_CH)
    
    C_model = np.zeros_like(C_measured, dtype=complex)
    k_local = np.arange(N_SUB) # Local indices [0, ..., 63]
    
    # 2. Reconstruct the channel model from the parameters
    for i in range(N_CH): # Loop over channels
        
        # Get distortion params for this channel
        alpha_i = alphas[i]
        phi_i = phis[i]
        
        # Global indices for this channel
        start_idx = i * N_SUB
        end_idx = (i + 1) * N_SUB
        k_global = np.arange(start_idx, end_idx)
        
        # Frequencies for these subcarriers
        freqs_hz = k_global * SUB_SPACING_HZ
        
        # --- Build the ideal channel part ---
        # C_ideal_slice = sum_l(g_l * exp(-j * 2 * pi * f_k * tau_l))
        C_ideal_slice = np.zeros(N_SUB, dtype=complex)
        for g_l, tau_l in zip(g, tau):
            C_ideal_slice += g_l * np.exp(-1j * 2 * np.pi * freqs_hz * tau_l)
        
        # --- Build the distortion part ---
        # D_i[k] = alpha_i * exp(-j * phi_i * k_local)
        distortion_array = alpha_i * np.exp(-1j * phi_i * k_local)
        
        # Apply distortion to get the modeled slice
        C_model[start_idx:end_idx] = C_ideal_slice * distortion_array
    
    # 3. Calculate and return the error
    # Sum of squared-magnitude of the difference
    error = np.sum(np.abs(C_measured - C_model)**2)
    return error

def reconstruct_global_channel(g_opt, tau_opt, N_SUB_TOTAL, SUB_SPACING_HZ):
    """
    Reconstructs the ideal channel model using the *optimized*
    multipath parameters (g_opt, tau_opt).
    """
    
    # Create an array of all subcarrier frequencies
    frequencies_hz = np.arange(N_SUB_TOTAL) * SUB_SPACING_HZ
    
    C_global_ideal = np.zeros(N_SUB_TOTAL, dtype=complex)
    
    # Sum the contributions from each *optimized* multipath component
    for g, tau in zip(g_opt, tau_opt):
        C_global_ideal += g * np.exp(-1j * 2 * np.pi * frequencies_hz * tau)
        
    return C_global_ideal

def calculate_and_print_metrics(C_ideal, C_measured, C_iterative, C_global, 
                                true_multipath, true_distortions,
                                g_opt, tau_opt, alphas_opt, phis_opt,
                                N_CHANNELS, N_PEAKS_GLOBAL):
    """
    Calculates and prints performance metrics comparing estimates to ground truth.
    
    Returns:
        tuple: Key performance metrics for plotting.
               (nmse_iterative_db, nmse_global_db, tau_err_ns, g_err_mag)
    """
    print("\n--- Performance Metrics ---")

    # 1. NMSE Calculation (Error vs. Ideal Channel)
    ideal_power = np.sum(np.abs(C_ideal)**2)
    
    # C_measured is now the noisy channel
    err_measured = np.sum(np.abs(C_measured - C_ideal)**2)
    err_iterative = np.sum(np.abs(C_iterative - C_ideal)**2)
    err_global = np.sum(np.abs(C_global - C_ideal)**2)
    
    # Add a small epsilon to ideal_power to prevent log10(0) if power is zero
    ideal_power += 1e-12
    
    nmse_measured_db = 10 * np.log10(err_measured / ideal_power)
    nmse_iterative_db = 10 * np.log10(err_iterative / ideal_power)
    nmse_global_db = 10 * np.log10(err_global / ideal_power)
    
    print("\n1. Normalized Mean Square Error (NMSE) vs. Ground Truth:")
    print(f"   Measured Channel (Distorted+Noise): {nmse_measured_db:8.3f} dB")
    print(f"   Iterative Approach:                 {nmse_iterative_db:8.3f} dB")
    print(f"   Global Approach:                    {nmse_global_db:8.3f} dB")
    print(f"   --------------------------------------------------------")
    print(f"   Iterative Improvement:              {nmse_measured_db - nmse_iterative_db:8.3f} dB")
    print(f"   Global Improvement:                 {nmse_measured_db - nmse_global_db:8.3f} dB")

    # 2. Distortion Parameter Error (Alpha, Phi)
    print("\n2. Distortion Parameter Estimation Error (Global Approach):")
    print("   Ch | True Alpha (Mag, Ph[deg]) | Est Alpha (Mag, Ph[deg]) | True Phi | Est Phi")
    print("   ----------------------------------------------------------------------------------")
    for i in range(N_CHANNELS):
        true_alpha, true_phi = true_distortions[i]
        est_alpha, est_phi = alphas_opt[i], phis_opt[i]
        
        true_alpha_mag = np.abs(true_alpha)
        true_alpha_ph = np.angle(true_alpha, deg=True)
        est_alpha_mag = np.abs(est_alpha)
        est_alpha_ph = np.angle(est_alpha, deg=True)
        
        print(f"   {i}  |  {true_alpha_mag:6.3f}, {true_alpha_ph:8.2f}   |  {est_alpha_mag:6.3f}, {est_alpha_ph:8.2f}   | {true_phi:8.4f} | {est_phi:8.4f}")

    # 3. Multipath Parameter Error (g, tau)
    print("\n3. Multipath Parameter Estimation Error (Global Approach):")
    print("   Path | True Gain (Mag, Ph[deg]) | Est Gain (Mag, Ph[deg]) | True Delay [ns] | Est Delay [ns]")
    print("   ---------------------------------------------------------------------------------------------")
    
    true_gains = np.array([g for g, d in true_multipath])
    true_delays = np.array([d for g, d in true_multipath])
    
    # Sort both true and estimated paths by delay for easier comparison
    true_sort_idx = np.argsort(true_delays)
    opt_sort_idx = np.argsort(tau_opt)
    
    true_g_sorted = true_gains[true_sort_idx]
    true_tau_sorted = true_delays[true_sort_idx]
    opt_g_sorted = g_opt[opt_sort_idx]
    opt_tau_sorted = tau_opt[opt_sort_idx]
    
    tau_err_ns = 0.0
    g_err_mag = 0.0
    num_paths_to_compare = 0

    # Handle case where no paths were generated
    if N_PEAKS_GLOBAL == 0:
        print("   (No paths generated or optimized)")
    else:
        # Ensure we don't index out of bounds if true/opt paths differ
        num_paths_to_compare = min(len(true_g_sorted), len(opt_g_sorted))

        for i in range(num_paths_to_compare):
            true_g_mag = np.abs(true_g_sorted[i])
            true_g_ph = np.angle(true_g_sorted[i], deg=True)
            true_t_ns = true_tau_sorted[i] * 1e9
            
            est_g_mag = np.abs(opt_g_sorted[i])
            est_g_ph = np.angle(opt_g_sorted[i], deg=True)
            est_t_ns = opt_tau_sorted[i] * 1e9
            
            print(f"   {i}    |   {true_g_mag:6.3f}, {true_g_ph:8.2f}   |  {est_g_mag:6.3f}, {est_g_ph:8.2f}   | {true_t_ns:13.2f} | {est_t_ns:13.2f}")
    
        # Calculate Mean Absolute Error for stats plot
        if num_paths_to_compare > 0:
            tau_err_ns = np.mean(np.abs(true_tau_sorted[:num_paths_to_compare] * 1e9 - opt_tau_sorted[:num_paths_to_compare] * 1e9))
            g_err_mag = np.mean(np.abs(np.abs(true_g_sorted[:num_paths_to_compare]) - np.abs(opt_g_sorted[:num_paths_to_compare])))
            print(f"\n   Mean Absolute Delay Error: {tau_err_ns:.2f} ns")
            print(f"   Mean Absolute Gain Mag Error: {g_err_mag:.3f}")

    print("-----------------------------------------------------------------------------------------------")
    
    return nmse_iterative_db, nmse_global_db, tau_err_ns, g_err_mag


def plot_channels(C_ideal, C_measured, C_iter_corrected, C_global_corrected, subcarriers_per_channel):
    """
    Plots the magnitude and phase of all channel versions.
    NOTE: This function is NO LONGER USED in interactive mode.
    The InteractiveSimulator class handles plotting directly.
    """
    print("Plotting results...")
    
    num_subcarriers_total = len(C_ideal)
    num_channels = num_subcarriers_total // subcarriers_per_channel
    k_global = np.arange(num_subcarriers_total)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # --- Plot 1: Magnitude ---
    ax1.plot(k_global, np.abs(C_ideal), label='Ideal Channel (Ground Truth)', color='blue', linestyle='--', linewidth=2.5)
    ax1.plot(k_global, np.abs(C_measured), label='Measured Stitched', color='red', alpha=0.6)
    if C_iter_corrected is not None:
        ax1.plot(k_global, np.abs(C_iter_corrected), label='Iteratively Corrected', color='green', alpha=0.7, linestyle=':')
    if C_global_corrected is not None:
        ax1.plot(k_global, np.abs(C_global_corrected), label='Globally Corrected (Model)', color='black', alpha=1.0, linestyle='-.')
        
    ax1.set_ylabel('Magnitude')
    ax1.set_title('WiFi Channel Stitching Simulation')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    # --- Plot 2: Phase ---
    ax2.plot(k_global, np.unwrap(np.angle(C_ideal)), label='Ideal Channel (Unwrapped Phase)', color='blue', linestyle='--', linewidth=2.5)
    ax2.plot(k_global, np.unwrap(np.angle(C_measured)), label='Measured Stitched (Unwrapped Phase)', color='red', alpha=0.6)
    if C_iter_corrected is not None:
        ax2.plot(k_global, np.unwrap(np.angle(C_iter_corrected)), label='Iteratively Corrected (Unwrapped Phase)', color='green', alpha=0.7, linestyle=':')
    if C_global_corrected is not None:
        ax2.plot(k_global, np.unwrap(np.angle(C_global_corrected)), label='Globally Corrected (Model)', color='black', alpha=1.0, linestyle='-.')
        
    ax2.set_xlabel('Global Subcarrier Index')
    ax2.set_ylabel('Phase (radians)')
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Add vertical lines for channel boundaries
    for i in range(1, num_channels):
        boundary = i * subcarriers_per_channel - 0.5 # Place between subcarriers
        ax1.axvline(x=boundary, color='black', linestyle=':', linewidth=1.5, label=f'Ch {i-1}/Ch {i} Boundary' if i == 1 else None)
        ax2.axvline(x=boundary, color='black', linestyle=':', linewidth=1.5)

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

# --- Interactive Simulator Class ---

class InteractiveSimulator:
    def __init__(self):
        # --- 1. Define Simulation Parameters ---
        self.N_CHANNELS = 4
        self.N_SUB_PER_CHAN = 64
        self.N_SUB_TOTAL = self.N_CHANNELS * self.N_SUB_PER_CHAN
        self.SUB_SPACING_HZ = 312.5e3 
        
        self.N_PEAKS_GLOBAL = 3 
        
        self.k_global = np.arange(self.N_SUB_TOTAL)

        # --- 2. Initial "Ground Truth" Values ---
        self.init_control_params = {
            'num_paths': 3,
            'delay_spread_ns': 200.0,
            'dist_mag': 0.1,
            'phase_jump_rad': np.pi / 4,
            'slope_mag': 0.01,
            'snr_db': 40.0,      # NEW: Default SNR
            'gap_size': 4        # NEW: Default Gap Size
        }
        
        # --- 3. Initial Distortion Values (Ch 0 is fixed) ---
        self.multipath_components = []
        self.distortion_params = []


        # --- 4. Setup the Figures and Axes ---
        self.fig_plots = plt.figure(figsize=(15, 8))
        self.fig_plots.canvas.manager.set_window_title('CFR Simulation')
        
        # --- MODIFIED --- Made control figure taller for new sliders
        self.fig_controls = plt.figure(figsize=(6, 7), facecolor='#f0f0f0') # Increased height
        self.fig_controls.canvas.manager.set_window_title('Controls')
        
        # --- NEW: Statistics Figure ---
        self.fig_stats = plt.figure(figsize=(10, 8), facecolor='#f0f0f0')
        self.fig_stats.canvas.manager.set_window_title('Performance Statistics')
        self.fig_stats.suptitle('Model Performance vs. Run', fontsize=16)
        
        
        self.fig_plots.suptitle('Interactive WiFi Channel Stitching Simulation', fontsize=16)
        
        self.ax1 = self.fig_plots.add_subplot(2, 1, 1)
        self.ax2 = self.fig_plots.add_subplot(2, 1, 2, sharex=self.ax1)
        self.fig_plots.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.92, hspace=0.1)

        # --- 5. Initialize Plot Lines (with nicer styles) ---
        self.line_ideal_mag, = self.ax1.plot(self.k_global, np.nan * self.k_global, label='Ideal (Ground Truth)', color='blue', linestyle='--', linewidth=2.5)
        # --- MODIFIED --- Measured line is now 'o' to show gaps
        self.line_meas_mag, = self.ax1.plot(self.k_global, np.nan * self.k_global, label='Measured Stitched (Noisy)', color='red', alpha=0.7, linestyle='None', marker='.', markersize=2)
        self.line_iter_mag, = self.ax1.plot(self.k_global, np.nan * self.k_global, label='Iteratively Corrected', color='green', alpha=0.8, linestyle=':', linewidth=2.0)
        self.line_global_mag, = self.ax1.plot(self.k_global, np.nan * self.k_global, label='Globally Corrected (Model)', color='black', alpha=1.0, linestyle='-.', linewidth=2.0)
        
        self.line_ideal_phase, = self.ax2.plot(self.k_global, np.nan * self.k_global, label='Ideal (Unwrapped Phase)', color='blue', linestyle='--', linewidth=2.5)
        # --- MODIFIED --- Measured line is now 'o' to show gaps
        self.line_meas_phase, = self.ax2.plot(self.k_global, np.nan * self.k_global, label='Measured (Unwrapped Phase)', color='red', alpha=0.7, linestyle='None', marker='.', markersize=2)
        self.line_iter_phase, = self.ax2.plot(self.k_global, np.nan * self.k_global, label='Iterative (Unwrapped Phase)', color='green', alpha=0.8, linestyle=':', linewidth=2.0)
        self.line_global_phase, = self.ax2.plot(self.k_global, np.nan * self.k_global, label='Global (Unwrapped Phase)', color='black', alpha=1.0, linestyle='-.', linewidth=2.0)

        self.ax1.set_ylabel('Magnitude')
        self.ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
        self.ax1.legend(loc='lower left', ncol=4)
        
        self.ax2.set_xlabel('Global Subcarrier Index')
        self.ax2.set_ylabel('Phase (radians)')
        self.ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

        # Add vertical lines for channel boundaries
        # --- MODIFIED: Store lines to update them later ---
        self.boundary_lines_mag = []
        self.boundary_lines_phase = []
        
        for i in range(1, self.N_CHANNELS):
            boundary = i * self.N_SUB_PER_CHAN - 0.5
            # Create two lines per boundary (for start/end of gap),
            # and initialize them at the same spot.
            line_mag_start = self.ax1.axvline(x=boundary, color='black', linestyle=':', linewidth=1.5)
            line_mag_end = self.ax1.axvline(x=boundary, color='black', linestyle=':', linewidth=1.5)
            self.boundary_lines_mag.extend([line_mag_start, line_mag_end])
            
            line_phase_start = self.ax2.axvline(x=boundary, color='black', linestyle=':', linewidth=1.5)
            line_phase_end = self.ax2.axvline(x=boundary, color='black', linestyle=':', linewidth=1.5)
            self.boundary_lines_phase.extend([line_phase_start, line_phase_end])
            
        # --- NEW: Setup Stats Plots ---
        self.ax_nmse = self.fig_stats.add_subplot(3, 1, 1)
        self.ax_nmse.set_title('NMSE vs. Ground Truth')
        self.ax_nmse.set_ylabel('NMSE (dB)')
        self.ax_nmse.grid(True)
        
        self.ax_tau_error = self.fig_stats.add_subplot(3, 1, 2)
        self.ax_tau_error.set_title('Path Parameter Error (Mean Absolute Error)')
        self.ax_tau_error.set_ylabel('Delay Error (ns)', color='tab:blue')
        self.ax_tau_error.grid(True)
        
        self.ax_g_error = self.ax_tau_error.twinx() # Share x-axis
        self.ax_g_error.set_ylabel('Gain Mag Error', color='tab:orange')
        
        self.ax_alpha_error = self.fig_stats.add_subplot(3, 1, 3)
        self.ax_alpha_error.set_title('Distortion Parameter Error (Mean Absolute Error)')
        self.ax_alpha_error.set_ylabel('Alpha Mag Error', color='tab:blue')
        self.ax_alpha_error.grid(True)
        
        self.ax_phi_error = self.ax_alpha_error.twinx()
        self.ax_phi_error.set_ylabel('Phi Slope Error', color='tab:orange')
        
        self.fig_stats.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.92, hspace=0.6)

        # --- NEW: History lists for stats ---
        self.step_history = []
        self.nmse_iterative_history = []
        self.nmse_global_history = []
        self.tau_error_history = []
        self.g_error_history = []
        self.alpha_error_history = []
        self.phi_error_history = []
        self.run_counter = 0

        # --- NEW: Plot lines for stats ---
        self.line_nmse_iterative, = self.ax_nmse.plot(self.step_history, self.nmse_iterative_history, 'g-o', label='Iterative NMSE')
        self.line_nmse_global, = self.ax_nmse.plot(self.step_history, self.nmse_global_history, 'k-o', label='Global NMSE')
        self.ax_nmse.legend(loc='upper right')
        
        self.line_tau_error, = self.ax_tau_error.plot(self.step_history, self.tau_error_history, 'b-o', label='Delay Error (ns)')
        self.line_g_error, = self.ax_g_error.plot(self.step_history, self.g_error_history, '-o', color='tab:orange', label='Gain Mag Error')
        
        self.line_alpha_error, = self.ax_alpha_error.plot(self.step_history, self.alpha_error_history, 'b-o', label='Alpha Mag Error')
        self.line_phi_error, = self.ax_phi_error.plot(self.step_history, self.phi_error_history, '-o', color='tab:orange', label='Phi Slope Error')

        # --- 6. Setup Sliders (on the control figure) ---
        self.sliders = {}
        
        slider_defs = {
            # Multipath
            'num_paths': (1, 5, self.init_control_params['num_paths'], 1, 'Num Paths'), 
            'delay_spread_ns': (20, 1000, self.init_control_params['delay_spread_ns'], 'RMS Delay (ns)'),
            # Distortions
            'dist_mag': (0.0, 0.5, self.init_control_params['dist_mag'], 'Distortion Mag'),
            'phase_jump_rad': (0.0, np.pi, self.init_control_params['phase_jump_rad'], 'Phase Jump (rad)'),
            'slope_mag': (0.0, 0.05, self.init_control_params['slope_mag'], 'Slope Mag'),
            # --- NEW SLIDERS ---
            'snr_db': (0, 60, self.init_control_params['snr_db'], 'SNR (dB)'),
            'gap_size': (0, 8, self.init_control_params['gap_size'], 1, 'Gap Size (Subcarriers)')
        }
        
        self.fig_controls.suptitle('Simulation Controls', fontsize=16)
        slider_keys = list(slider_defs.keys())
        
        y_pos = 0.95 # Start higher
        for i, key in enumerate(slider_keys):
            ax = self.fig_controls.add_axes([0.15, y_pos, 0.7, 0.03])
            
            if key == 'num_paths' or key == 'gap_size':
                valmin, valmax, valinit, step, label = slider_defs[key]
                self.sliders[key] = Slider(ax, label, valmin, valmax, valinit=valinit, valstep=step)
            else:
                valmin, valmax, valinit, label = slider_defs[key]
                self.sliders[key] = Slider(ax, label, valmin, valmax, valinit=valinit)
            
            # --- MODIFIED --- Tighter spacing
            y_pos -= 0.1 

        # --- 7. Setup Buttons (on the control figure) ---
        ax_btn_channel = self.fig_controls.add_axes([0.1, y_pos - 0.0, 0.8, 0.05])
        self.btn_rand_channel = Button(ax_btn_channel, 'Randomize Channel')
        self.btn_rand_channel.on_clicked(self.on_randomize_channel)
        
        ax_btn_dist = self.fig_controls.add_axes([0.1, y_pos - 0.07, 0.8, 0.05])
        self.btn_rand_dist = Button(ax_btn_dist, 'Randomize Distortions')
        self.btn_rand_dist.on_clicked(self.on_randomize_distortions)
        
        ax_button = self.fig_controls.add_axes([0.1, y_pos - 0.16, 0.8, 0.06])
        self.btn_run_opt = Button(ax_button, 'Run Global Optimization')
        self.btn_run_opt.on_clicked(self.run_global_optimization)
        
        # --- NEW: Clear Stats Button ---
        ax_btn_clear = self.fig_controls.add_axes([0.1, y_pos - 0.23, 0.8, 0.05])
        self.btn_clear_stats = Button(ax_btn_clear, 'Clear Stats History')
        self.btn_clear_stats.on_clicked(self.on_clear_stats)


        # --- 8. Run Initial Calculation ---
        self.generate_new_channel()
        self.generate_new_distortions()
        self.fast_update(None)

    # --- NEW CALLBACK FUNCTIONS ---
    
    def on_clear_stats(self, event):
        """Clears the statistics plot history."""
        print("\n--- Clearing Statistics History ---")
        self.step_history.clear()
        self.nmse_iterative_history.clear()
        self.nmse_global_history.clear()
        self.tau_error_history.clear()
        self.g_error_history.clear()
        self.alpha_error_history.clear()
        self.phi_error_history.clear()
        self.run_counter = 0
        
        # Update plot lines with empty data
        self.line_nmse_iterative.set_data([], [])
        self.line_nmse_global.set_data([], [])
        self.line_tau_error.set_data([], [])
        self.line_g_error.set_data([], [])
        self.line_alpha_error.set_data([], [])
        self.line_phi_error.set_data([], [])
        
        # Rescale axes
        for ax in [self.ax_nmse, self.ax_tau_error, self.ax_alpha_error]:
            ax.relim()
            ax.autoscale_view()
        self.ax_g_error.relim()
        self.ax_g_error.autoscale_view()
        self.ax_phi_error.relim()
        self.ax_phi_error.autoscale_view()
            
        self.fig_stats.canvas.draw_idle()

    
    def generate_new_channel(self):
        """Generates a new set of multipath components."""
        num_paths = int(self.sliders['num_paths'].val)
        delay_spread_ns = self.sliders['delay_spread_ns'].val
        
        self.N_PEAKS_GLOBAL = num_paths # Update class variable
        
        print(f"\n--- Generating New Channel: {num_paths} paths, {delay_spread_ns:.0f} ns delay spread ---")

        # Use an exponential delay profile and Rayleigh fading
        # This is a more realistic channel model
        
        self.multipath_components = []
        
        # 1. First path (often line-of-sight)
        # Add a small base delay
        base_delay_sec = 50e-9 
        gain1 = 1.0 # Reference gain
        self.multipath_components.append((gain1, base_delay_sec))
        
        if num_paths > 1:
            # 2. Subsequent paths (NLOS)
            
            # Generate delays from an exponential distribution
            # The 'mean' of the *additional* delay is the RMS delay spread
            mean_delay_sec = delay_spread_ns * 1e-9
            additional_delays = np.random.exponential(mean_delay_sec, num_paths - 1)
            
            # Total delays are base + additional
            total_delays_sec = base_delay_sec + additional_delays
            
            # Generate gains
            # Power decays exponentially with delay
            # Gain is complex (Rayleigh magnitude, Uniform phase)
            for delay in total_delays_sec:
                # Calculate avg power for this delay relative to first path
                # P(t) = P0 * exp(-t / T_rms)
                avg_power = np.exp(-(delay - base_delay_sec) / mean_delay_sec)
                
                # Magnitude is Rayleigh distributed, scale factor is related to avg_power
                # E[Mag^2] = 2 * scale^2 = avg_power => scale = sqrt(avg_power / 2)
                scale = np.sqrt(avg_power / 2.0)
                magnitude = np.random.rayleigh(scale)
                
                # Phase is uniform from -pi to pi
                phase = np.random.uniform(-np.pi, np.pi)
                
                gain = magnitude * np.exp(1j * phase)
                self.multipath_components.append((gain, delay))

        # Sort by delay
        self.multipath_components.sort(key=lambda x: x[1])

    def generate_new_distortions(self):
        """Generates a new set of distortion parameters."""
        dist_mag = self.sliders['dist_mag'].val
        phase_jump_rad = self.sliders['phase_jump_rad'].val
        slope_mag = self.sliders['slope_mag'].val
        
        print("\n--- Generating New Distortions ---")
        
        self.distortion_params = []
        # Ch 0 is reference
        self.distortion_params.append((1.0, 0.0))
        
        for i in range(1, self.N_CHANNELS):
            # Random magnitude distortion
            mag = np.random.uniform(1.0 - dist_mag, 1.0 + dist_mag)
            
            # Random phase jump
            phase = np.random.uniform(-phase_jump_rad, phase_jump_rad)
            
            alpha_i = mag * np.exp(1j * phase)
            
            # Random slope
            phi_i = np.random.uniform(-slope_mag, slope_mag)
            
            self.distortion_params.append((alpha_i, phi_i))
            
    def on_randomize_channel(self, event):
        """Called when 'Randomize Channel' button is clicked."""
        self.generate_new_channel()
        self.fast_update(None)

    def on_randomize_distortions(self, event):
        """Called when 'Randomize Distortions' button is clicked."""
        self.generate_new_distortions()
        self.fast_update(None)
    
    # --- REMOVED old on_slider_change and on_textbox_submit ---

    def fast_update(self, val):
        """
        Runs on button clicks or slider changes.
        Updates Ideal, Measured, and Iterative plots.
        """
        
        # 1. Re-run ideal simulation
        self.C_ideal = simulate_ideal_channel(self.N_SUB_TOTAL, self.SUB_SPACING_HZ, self.multipath_components)
        
        # 2. Apply distortions
        self.C_measured_stitched = apply_channel_distortions(self.C_ideal, self.N_CHANNELS, self.N_SUB_PER_CHAN, self.distortion_params)
        
        # --- NEW: 3. Add AWGN ---
        snr_db = self.sliders['snr_db'].val
        snr_linear = 10**(snr_db / 10.0)
        
        signal_power = np.mean(np.abs(self.C_measured_stitched)**2)
        noise_power = signal_power / snr_linear
        # Add epsilon to prevent divide by zero if signal power is 0
        noise_variance = (noise_power / 2.0) + 1e-12 
        
        noise = (np.random.randn(self.N_SUB_TOTAL) + 1j * np.random.randn(self.N_SUB_TOTAL)) * np.sqrt(noise_variance)
        
        # This is the signal that the algorithms will actually see
        self.C_measured_with_noise = self.C_measured_stitched + noise
        
        # 4. Run iterative correction on the *noisy* data
        self.C_iterative_corrected, self.global_gammas = correct_iterative_approach(
            self.C_measured_with_noise, self.N_CHANNELS, self.N_SUB_PER_CHAN
        )
        
        # --- NEW: 5. Create gapped channel *for plotting only* ---
        gap_size = int(self.sliders['gap_size'].val)
        self.C_measured_for_plotting = np.copy(self.C_measured_with_noise)
        
        if gap_size > 0:
            for i in range(1, self.N_CHANNELS):
                boundary_idx = i * self.N_SUB_PER_CHAN
                start_gap = max(boundary_idx - gap_size // 2, 0)
                end_gap = min(boundary_idx + (gap_size - gap_size // 2), self.N_SUB_TOTAL)
                self.C_measured_for_plotting[start_gap:end_gap] = np.nan
        
        
        # --- NEW: 6. Update boundary line positions based on gap_size ---
        for i in range(1, self.N_CHANNELS):
            boundary_idx = i * self.N_SUB_PER_CHAN
            
            if gap_size == 0:
                # Special case: show a single line at the boundary
                start_pos = boundary_idx - 0.5
                end_pos = boundary_idx - 0.5
            else:
                # Show two lines at the edges of the gap
                start_gap = max(boundary_idx - gap_size // 2, 0)
                end_gap = min(boundary_idx + (gap_size - gap_size // 2), self.N_SUB_TOTAL)
                
                # -0.5 to be between subcarriers
                start_pos = start_gap - 0.5
                # -1.5 to be *before* the last NaN subcarrier's index
                end_pos = end_gap - 1.5 
            
            # Get the correct pair of lines to update
            line_idx = (i - 1) * 2
            
            self.boundary_lines_mag[line_idx].set_xdata([start_pos, start_pos])
            self.boundary_lines_mag[line_idx + 1].set_xdata([end_pos, end_pos])
            
            self.boundary_lines_phase[line_idx].set_xdata([start_pos, start_pos])
            self.boundary_lines_phase[line_idx + 1].set_xdata([end_pos, end_pos])


        # 7. Update plot lines (was step 6)
        self.line_ideal_mag.set_ydata(np.abs(self.C_ideal))
        self.line_meas_mag.set_ydata(np.abs(self.C_measured_for_plotting))
        self.line_iter_mag.set_ydata(np.abs(self.C_iterative_corrected))
        
        self.line_ideal_phase.set_ydata(np.unwrap(np.angle(self.C_ideal)))
        self.line_meas_phase.set_ydata(np.unwrap(np.angle(self.C_measured_for_plotting)))
        self.line_iter_phase.set_ydata(np.unwrap(np.angle(self.C_iterative_corrected)))
        
        # 8. Clear the global line (it's now out of date) (was step 7)
        self.line_global_mag.set_ydata(np.nan * self.k_global)
        self.line_global_phase.set_ydata(np.nan * self.k_global)
        
        # 9. Redraw the plot figure (was step 8)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig_plots.canvas.draw_idle()

    def run_global_optimization(self, event):
        """
        Runs on button click.
        Performs the heavy optimization and updates the global plot/metrics.
        """
            
        print(f"\n--- Running Global Approach (Optimization for {self.N_PEAKS_GLOBAL} paths) ---")
        
        # --- MODIFIED: All initial guesses are now derived from the *noisy* channel ---
        print("Bootstrapping initial guesses from noisy channel...")
        
        # 1. Get a *good* phi_0 guess (slopes)
        # Use the noisy measured channel and the noisy iterative channel
        phase_diff = np.unwrap(np.angle(self.C_measured_with_noise)) - np.unwrap(np.angle(self.C_iterative_corrected))
        phi_slopes_initial = estimate_slopes_from_phase_diff(
            phase_diff, self.N_CHANNELS, self.N_SUB_PER_CHAN
        )
        print(f"  Initial phi guesses (slopes): {[f'{p:.4f}' for p in phi_slopes_initial]}")

        # 2. Get a *good* alpha_0 (gains) and g_0/tau_0 (paths) guess
        # First, "de-slope" the *noisy* measured channel
        C_desloped = np.zeros_like(self.C_measured_with_noise)
        k_local = np.arange(self.N_SUB_PER_CHAN)
        for i in range(self.N_CHANNELS):
            start_idx = i * self.N_SUB_PER_CHAN
            end_idx = (i + 1) * self.N_SUB_PER_CHAN
            
            deslope_array = np.exp(1j * phi_slopes_initial[i] * k_local)
            
            C_desloped[start_idx:end_idx] = self.C_measured_with_noise[start_idx:end_idx] * deslope_array
            
        # Run iterative correction on this de-sloped *noisy* channel
        C_iterative_v2, alphas_initial = correct_iterative_approach(
            C_desloped, self.N_CHANNELS, self.N_SUB_PER_CHAN
        )
        print(f"  Initial alpha guesses (mags): {[f'{np.abs(a):.3f}' for a in alphas_initial]}")
        
        # Get path guesses from this new, cleaner (but still noisy) channel
        g_0, tau_0 = initialize_global_approach(
            C_iterative_v2, self.SUB_SPACING_HZ, self.N_PEAKS_GLOBAL
        )
        
        # ----------------------------------------
        
        # 3. Pack and run optimizer
        x0 = pack_parameters(g_0, tau_0, alphas_initial, phi_slopes_initial)
        # --- MODIFIED: Pass the *noisy* channel to the optimizer
        obj_args = (self.C_measured_with_noise, self.N_CHANNELS, self.N_SUB_PER_CHAN, self.N_PEAKS_GLOBAL, self.SUB_SPACING_HZ)
        
        options = {'maxfun': 50000, 'maxiter': 100000}
        result = minimize(global_objective_function, x0, args=obj_args, method='L-BFGS-B', options=options)
        
        print(f"Optimization finished: {result.message}")
        
        g_1 = np.zeros_like(g_0)
        tau_1 = np.zeros_like(tau_0)
        alphas_initial_1 = np.zeros_like(alphas_initial)
        phi_slopes_initial_1 = np.zeros_like(phi_slopes_initial)
        x1 = pack_parameters(g_1, tau_1, alphas_initial_1, phi_slopes_initial_1)
        result_1 = minimize(global_objective_function, x1, args=obj_args, method='L-BFGS-B', options=options)

        # 4. Unpack and reconstruct
        x_opt = result.x
        x_opt_1 = result_1.x
        g_opt, tau_opt, alphas_opt, phis_opt = unpack_parameters(x_opt, self.N_PEAKS_GLOBAL, self.N_CHANNELS)
        g_opt_1, tau_opt_1, alphas_opt_1, phis_opt_1 = unpack_parameters(x_opt_1, self.N_PEAKS_GLOBAL, self.N_CHANNELS) 
        print("Reconstructing globally optimized channel...")
        C_global_corrected = reconstruct_global_channel(g_opt, tau_opt, self.N_SUB_TOTAL, self.SUB_SPACING_HZ)
        C_global_corrected_1 = reconstruct_global_channel(g_opt_1, tau_opt_1, self.N_SUB_TOTAL, self.SUB_SPACING_HZ)
        # 5. Update the global line on the plot
        self.line_global_mag.set_ydata(np.abs(C_global_corrected))
        self.line_global_phase.set_ydata(np.unwrap(np.angle(C_global_corrected)))
        self.fig_plots.canvas.draw_idle()
        self.line_global_mag_1.set_ydata(np.abs(C_global_corrected_1))
        self.line_global_phase_1.set_ydata(np.unwrap(np.angle(C_global_corrected_1)))
        self.fig_plots.canvas_1.draw_idle()

        # 6. Print metrics to console AND GET VALUES FOR PLOTTING
        nmse_iter, nmse_glob, tau_err, g_err = calculate_and_print_metrics(
            self.C_ideal, self.C_measured_with_noise, self.C_iterative_corrected, C_global_corrected,
            self.multipath_components, self.distortion_params,
            g_opt, tau_opt, alphas_opt, phis_opt,
            self.N_CHANNELS, self.N_PEAKS_GLOBAL
        )
        
        # --- NEW: 7. Update Statistics Plot ---
        self.run_counter += 1
        self.step_history.append(self.run_counter)
        self.nmse_iterative_history.append(nmse_iter)
        self.nmse_global_history.append(nmse_glob)
        self.tau_error_history.append(tau_err)
        self.g_error_history.append(g_err)
        
        # Calculate alpha and phi errors
        true_alphas = np.array([a[0] for a in self.distortion_params])
        true_phis = np.array([a[1] for a in self.distortion_params])
        alpha_err = np.mean(np.abs(np.abs(true_alphas) - np.abs(alphas_opt)))
        phi_err = np.mean(np.abs(true_phis - phis_opt))
        
        self.alpha_error_history.append(alpha_err)
        self.phi_error_history.append(phi_err)
        
        # Update plot data
        self.line_nmse_iterative.set_data(self.step_history, self.nmse_iterative_history)
        self.line_nmse_global.set_data(self.step_history, self.nmse_global_history)
        
        self.line_tau_error.set_data(self.step_history, self.tau_error_history)
        self.line_g_error.set_data(self.step_history, self.g_error_history)
        
        self.line_alpha_error.set_data(self.step_history, self.alpha_error_history)
        self.line_phi_error.set_data(self.step_history, self.phi_error_history)
        
        # Rescale axes
        for ax in [self.ax_nmse, self.ax_tau_error, self.ax_alpha_error]:
            ax.relim()
            ax.autoscale_view()
        self.ax_g_error.relim()
        self.ax_g_error.autoscale_view()
        self.ax_phi_error.relim()
        self.ax_phi_error.autoscale_view()
            
        self.fig_stats.canvas.draw_idle()


# --- Main simulation ---
if __name__ == "__main__":
    
    # All the simulation logic is now inside the class.
    # We just need to create an instance and show the plot.
    
    print("Starting interactive simulation...")
    print("Adjust sliders and click 'Randomize' buttons to create a new channel scenario.")
    print("Click 'Run Global Optimization' to solve for the current scenario.")
    print("A third 'Performance Statistics' window will show history.")
    
    sim = InteractiveSimulator()
    plt.show() # This blocks and keeps the interactive window open