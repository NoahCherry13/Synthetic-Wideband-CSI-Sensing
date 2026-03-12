import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Parameter Packing & Normalization
# ==========================================
# CRITICAL FIX: We scale the parameters so the optimizer sees values near O(1).
# We optimize tau in nanoseconds, so x_tau = 50 means 50ns.

def unpack_scaled_parameters(x, N_PEAKS, N_CH):
    T, I = N_PEAKS, N_CH
    # 1. Complex Gains (Real and Imaginary parts)
    g_real = x[0:T]
    g_imag = x[T:2*T]
    g = g_real + 1j * g_imag
    
    # 2. Time Delays (Optimizer sees ns, we convert to seconds for physics)
    tau = x[2*T : 3*T] * 1e-9 
    
    # 3. Hardware Gains (Channel 0 is reference, so we only estimate I-1 channels)
    alphas_real = x[3*T : 3*T + (I-1)]
    alphas_imag = x[3*T + (I-1) : 3*T + 2*(I-1)]
    alphas = np.concatenate([[1.0 + 0j], alphas_real + 1j * alphas_imag])
    
    # 4. Hardware Phase (Channel 0 is reference)
    phis = np.concatenate([[0.0], x[3*T + 2*(I-1) : 3*T + 3*(I-1)]])
    
    return g, tau, alphas, phis

def generate_channel(x, N_CH, N_SUB, N_PEAKS, SUB_SPACING_HZ):
    """Generates the stitched channel based on the current parameter guess."""
    g, tau, alphas, phis = unpack_scaled_parameters(x, N_PEAKS, N_CH)
    C_model = np.zeros(N_CH * N_SUB, dtype=complex)
    
    # Subcarrier index relative to the channel block
    k_local = np.arange(N_SUB) 
    
    for i in range(N_CH):
        start_idx = i * N_SUB
        end_idx = start_idx + N_SUB
        
        # Absolute frequency for this channel's subcarriers
        freqs_hz = np.arange(start_idx, end_idx) * SUB_SPACING_HZ
        
        # 1. Ideal Multipath Channel (Sum of delayed paths)
        C_ideal = np.zeros(N_SUB, dtype=complex)
        for g_l, tau_l in zip(g, tau):
            C_ideal += g_l * np.exp(-1j * 2 * np.pi * freqs_hz * tau_l)
            
        # 2. Apply Hardware Distortion (Stitching offset)
        # Note: In your thesis, phase is e^{-j * phi * n}
        distortion = alphas[i] * np.exp(-1j * phis[i] * k_local)
        
        C_model[start_idx:end_idx] = C_ideal * distortion
        
    return C_model

def get_real_residual(x, C_measured, *args):
    """Computes the error between measured and modeled data."""
    C_model = generate_channel(x, *args)
    residual_complex = C_measured - C_model
    # Optimizers need real numbers; split complex residual into real/imag concatenated
    return np.concatenate([np.real(residual_complex), np.imag(residual_complex)])

# ==========================================
# 2. Improved Robust SCA (Levenberg-Marquardt)
# ==========================================

def compute_jacobian(x, residual_func, *args):
    """Numerically calculates the Jacobian using central differences for stability."""
    r0 = residual_func(x, *args)
    J = np.zeros((len(r0), len(x)))
    eps = 1e-6 # Stable because parameters are now normalized to O(1)
    
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        r_plus = residual_func(x_plus, *args)
        r_minus = residual_func(x_minus, *args)
        J[:, i] = (r_plus - r_minus) / (2 * eps)
        
    return J, r0

def sca_optimize(x0, C_measured, args, max_iter=1000, lambda_reg=1.0):
    x_k = x0.copy()
    history = []
    
    for iteration in range(max_iter):
        J, r_k = compute_jacobian(x_k, get_real_residual, C_measured, *args)
        error = np.sum(r_k**2)
        history.append(error)
        
        # Formulate convex subproblem
        J_T_J = J.T @ J
        # Scale the regularization by the diagonal of J^T J for better conditioning
        diag_reg = lambda_reg * np.diag(np.diag(J_T_J) + 1e-8)
        A = J_T_J + diag_reg
        
        # [CRITICAL FIX]: The target vector 'b' must point towards the negative residual
        b = -J.T @ r_k 
        
        try:
            delta_x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered, increasing damping.")
            lambda_reg *= 10
            continue
            
        x_new = x_k + delta_x
        r_new = get_real_residual(x_new, C_measured, *args)
        new_error = np.sum(r_new**2)
        
        # Trust Region / Damping Update
        if new_error < error:
            # Step accepted! The error went down.
            x_k = x_new
            lambda_reg = max(1e-7, lambda_reg / 5.0) # Shrink penalty to take bigger steps
            
            # Print progress every 5 successful iterations
            if len(history) % 5 == 0:
                print(f"Iter {iteration:02d} | Error: {new_error:.4f} | Lambda: {lambda_reg:.1e}")
                
            if np.linalg.norm(delta_x) < 1e-5:
                print(f"Converged beautifully at iteration {iteration}")
                break
        else:
            # Step rejected! The linear approximation was poor.
            lambda_reg *= 5.0 # Increase penalty to take a smaller, safer step next time
            # print(f"  -> Step rejected at iter {iteration}. Increasing lambda to {lambda_reg:.1e}")
            
    return x_k, history

# ==========================================
# 3. Execution & Validation
# ==========================================
if __name__ == "__main__":
    N_CH = 3
    N_SUB = 64
    N_PEAKS = 2
    SUB_SPACING_HZ = 312.5e3
    args = (N_CH, N_SUB, N_PEAKS, SUB_SPACING_HZ)
    
    # --- 1. Generate True Data ---
    # Gains: [Real1, Real2, Imag1, Imag2]
    # Taus: [Path1_ns, Path2_ns]
    # Alphas (CH1, CH2): [Real, Real, Imag, Imag]
    # Phis (CH1, CH2): [Rad, Rad]
    x_true = np.array([
        0.8, 0.4, 0.1, -0.2,       # Gains (complex)
        10.0, 25.0,                # Taus (in nanoseconds)
        0.95, 1.05, 0.05, -0.05,   # Alphas (complex)
        0.2, -0.1                  # Phis (radians)
    ])
    
    print("Generating ground truth channel...")
    C_true = generate_channel(x_true, *args)
    
    # Add noise (SNR ~ 20dB)
    noise_power = 0.05
    noise = (np.random.randn(len(C_true)) + 1j * np.random.randn(len(C_true))) * noise_power
    C_measured = C_true + noise
    
    # --- 2. Optimize ---
    # Provide a reasonable initial guess (flat channel, zero distortion, slight delays)
    x0 = np.zeros(len(x_true))
    x0[0:N_PEAKS] = 0.5            # Guess real gains
    x0[2*N_PEAKS : 3*N_PEAKS] = [5.0, 20.0] # Guess delays in ns
    x0[3*N_PEAKS : 3*N_PEAKS + (N_CH-1)] = 1.0 # Guess hardware gain = 1.0
    
    print("Starting optimization...")
    x_opt, error_hist = sca_optimize(x0, C_measured, args, max_iter=10000)
    
    # --- 3. Results ---
    print("\n--- Parameter Estimation Results ---")
    g_true, tau_true, a_true, p_true = unpack_scaled_parameters(x_true, N_PEAKS, N_CH)
    g_opt, tau_opt, a_opt, p_opt = unpack_scaled_parameters(x_opt, N_PEAKS, N_CH)
    
    '''
        0.8, 0.4, 0.1, -0.2,       # Gains (complex)
        10.0, 45.0,                # Taus (in nanoseconds)
        0.95, 1.05, 0.05, -0.05,   # Alphas (complex)
        0.2, -0.1                  # Phis (radians)
    '''

    print(f"True Taus (ns): {tau_true * 1e9}")
    print(f"Est. Taus (ns): {tau_opt * 1e9}")
    print(f"True Alphas   : {a_true}")
    print(f"Est. Alphas   : {np.round(a_opt, 3)}")
    
    # --- 4. Plotting ---
    C_opt = generate_channel(x_opt, *args)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Convergence
    ax1.plot(error_hist, marker='o', color='blue')
    ax1.set_title("Optimization Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Sum of Squared Errors")
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # Plot 2: Channel Match
    ax2.plot(np.abs(C_measured), label="Measured (Noisy)", alpha=0.5)
    ax2.plot(np.abs(C_opt), label="Estimated Fit", color='red', linestyle='--')
    ax2.set_title("Channel Magnitude Over Subcarriers")
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()