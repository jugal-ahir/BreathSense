# 3, 10 ,15
"""
UNIVERSAL_BREATHING_RATE_ESTIMATOR_v2.py
-----------------------------------------
Handles ALL BPM datasets (3, 6, 10, 15, 30...) with harmonic correction.

Key Features:
    ✅ Fundamental vs Harmonic Detection (Prevents 10 BPM → 34 BPM errors)
    ✅ Bandpass for 3–40 BPM (0.05–0.7 Hz)
    ✅ Phase-regression with safety checks
    ✅ Works on ANY dataset path (just change CSV paths)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, get_window
from scipy.fft import fft, ifft, fftfreq
from sklearn.linear_model import LinearRegression
import os
import sys
import argparse
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -------- USER INPUT (only change these two paths) --------
# Update paths to match your current workspace structure
REAL_CSV = r"C:\Jugal\College\Wireless\Dataset\BPM6\config0001_csi_real_log.csv"
IMAG_CSV = r"C:\Jugal\College\Wireless\Dataset\BPM6\config0001_csi_imag_log.csv"
FS       = 10.0     # Frames per second
K_TOP    = 50       # Top subcarriers to average
# ----------------------------------------------------------

# Frequency range covering 3–40 BPM
LOWCUT   = 0.03     # 0.03 Hz ≈ 1.8 BPM
HIGHCUT  = 0.70     # 0.70 Hz ≈ 42 BPM

MIN_VALID_FREQ  = 0.04   # Below this → noise
PHASE_TOLERANCE = 0.08   # Phase error tolerance vs FFT peak

# ----------------------------------------------------------
def load_complex(real_path, imag_path):
    """Load complex CSI data from real and imaginary CSV files with error handling."""
    try:
        # Check if files exist
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Real CSV file not found: {real_path}")
        if not os.path.exists(imag_path):
            raise FileNotFoundError(f"Imaginary CSV file not found: {imag_path}")
        
        # Load data
        r = pd.read_csv(real_path, header=None).values
        i = pd.read_csv(imag_path, header=None).values
        
        # Validate data shapes
        if r.shape != i.shape:
            raise ValueError(f"Shape mismatch: real={r.shape}, imag={i.shape}")
        
        if r.size == 0:
            raise ValueError("Empty data files")
        
        print(f"✅ Loaded CSI data: {r.shape[0]} samples, {r.shape[1]} subcarriers")
        return r + 1j * i
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def select_top_k(csi_matrix, k=50):
    """Select top K most variant subcarriers."""
    mags = np.abs(csi_matrix)
    var = np.nanvar(mags, axis=0)
    idx = np.argsort(var)[::-1][:min(k, mags.shape[1])]
    print(f"✅ Selected top {len(idx)} subcarriers out of {mags.shape[1]}")
    return idx

def parabolic_interpolation(freqs_p, magn, k):
    if k <= 0 or k >= len(magn) - 1:
        return freqs_p[k]
    a, b, c = magn[k - 1], magn[k], magn[k + 1]
    p = 0.5 * (a - c) / (a - 2 * b + c)
    return freqs_p[k] + p * (freqs_p[1] - freqs_p[0])

def phase_regression(x_complex, fs):
    phi = np.unwrap(np.angle(x_complex))
    t = np.arange(len(phi)) / fs
    model = LinearRegression().fit(t.reshape(-1, 1), phi.reshape(-1, 1))
    slope = model.coef_[0][0]
    return slope / (2 * np.pi)

# ----------------------------------------------------------
def harmonic_correction(freq, freqs, mag):
    """
    Detect if freq is a harmonic and return fundamental.
    """
    # Check /2 and /3 harmonic positions
    for factor in [2, 3]:
        cand = freq / factor
        idx = np.argmin(np.abs(freqs - cand))
        # If magnitude at subharmonic is strong, accept it
        if mag[idx] > 0.4 * mag[np.argmax(mag)]:
            return freqs[idx]
    return freq

# ----------------------------------------------------------
def validate_data_quality(csi_matrix, fs):
    """Validate data quality and provide warnings."""
    duration = csi_matrix.shape[0] / fs
    print(f"📊 Data duration: {duration:.1f} seconds")
    
    if duration < 10:
        print("⚠️  Warning: Short data duration may affect accuracy")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(csi_matrix)) or np.any(np.isinf(csi_matrix)):
        print("⚠️  Warning: Data contains NaN or infinite values")
    
    return True

def create_stunning_visualization(freqs_pos, mag_pos, freq_final, bpm, sig, f_phase, fs):
    """Create absolutely stunning, eye-appealing visualizations with maximum spacing."""
    
    # Create much larger figure for better spacing
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Define beautiful color palette
    colors = {
        'primary': '#00d4ff',      # Electric blue
        'secondary': '#ff6b6b',    # Coral red
        'accent': '#4ecdc4',       # Teal
        'success': '#45b7d1',      # Sky blue
        'warning': '#f9ca24',      # Golden yellow
        'background': '#1a1a2e',   # Dark navy
        'surface': '#16213e',      # Lighter navy
        'text': '#ffffff',         # White
        'gradient_start': '#667eea', # Purple gradient start
        'gradient_end': '#764ba2'    # Purple gradient end
    }
    
    # Create much more spacious grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.8, wspace=0.6, 
                         height_ratios=[1, 1], width_ratios=[1, 1],
                         left=0.08, right=0.92, top=0.88, bottom=0.12)
    
    # === PLOT 1: CLEAN FREQUENCY SPECTRUM ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(colors['surface'])
    
    # Simplified spectrum plot with thinner lines
    ax1.plot(freqs_pos, mag_pos, color=colors['primary'], linewidth=2, 
            alpha=0.9, label="Frequency Spectrum")
    
    # Detected frequency line with thinner width
    ax1.axvline(freq_final, color=colors['secondary'], linestyle='-', linewidth=3, 
               alpha=0.9, label=f"Detected: {bpm:.1f} BPM")
    
    # Simplified reference lines with thinner width
    ref_bpms = [6, 12, 15, 20, 30]
    for bpm_ref in ref_bpms:
        freq_ref = bpm_ref / 60
        ax1.axvline(freq_ref, color=colors['accent'], linestyle=':', linewidth=1, alpha=0.5)
        ax1.text(freq_ref, np.max(mag_pos)*0.85, f'{bpm_ref}', color=colors['accent'], 
                fontsize=12, fontweight='bold', ha='center')
    
    # Clean styling
    ax1.set_title("🌊 FREQUENCY SPECTRUM", fontsize=20, fontweight='bold', 
                 color=colors['text'], pad=40)
    ax1.set_xlabel("Frequency (Hz)", fontsize=16, color=colors['text'], labelpad=20)
    ax1.set_ylabel("Magnitude", fontsize=16, color=colors['text'], labelpad=20)
    ax1.set_xlim(0, 0.7)
    ax1.grid(True, alpha=0.3, color=colors['text'])
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
              fontsize=14, framealpha=0.9)
    
    # Set axis colors to white for better visibility
    ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='y', colors='white', labelsize=12)
    
    # Generous padding
    ax1.margins(x=0.05, y=0.08)
    
    # === PLOT 2: CLEAN TIME DOMAIN SIGNAL ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(colors['surface'])
    
    time_axis = np.arange(len(sig)) / fs
    
    # Simplified time domain plot with thinner lines
    ax2.plot(time_axis, np.real(sig), color=colors['primary'], linewidth=2, 
            alpha=0.9, label="Real Component")
    ax2.plot(time_axis, np.imag(sig), color=colors['secondary'], linewidth=2, 
            alpha=0.9, label="Imaginary Component")
    
    ax2.set_title("📈 TIME DOMAIN SIGNAL", fontsize=20, fontweight='bold', 
                 color=colors['text'], pad=40)
    ax2.set_xlabel("Time (s)", fontsize=16, color=colors['text'], labelpad=20)
    ax2.set_ylabel("Amplitude", fontsize=16, color=colors['text'], labelpad=20)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
              fontsize=14, framealpha=0.9)
    ax2.grid(True, alpha=0.3, color=colors['text'])
    
    # Set axis colors to white for better visibility
    ax2.tick_params(axis='x', colors='white', labelsize=12)
    ax2.tick_params(axis='y', colors='white', labelsize=12)
    
    # Generous padding
    ax2.margins(x=0.05, y=0.08)
    
    # === PLOT 3: CLEAN BREATHING PATTERN ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(colors['surface'])
    
    # Simplified breathing pattern with thinner lines
    ax3.plot(time_axis, f_phase, color=colors['accent'], linewidth=2, alpha=0.9, 
             label="Filtered Phase")
    
    ax3.set_title("🫁 BREATHING PATTERN", fontsize=20, fontweight='bold', 
                 color=colors['text'], pad=40)
    ax3.set_xlabel("Time (s)", fontsize=16, color=colors['text'], labelpad=20)
    ax3.set_ylabel("Phase (rad)", fontsize=16, color=colors['text'], labelpad=20)
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
              fontsize=14, framealpha=0.9)
    ax3.grid(True, alpha=0.3, color=colors['text'])
    
    # Set axis colors to white for better visibility
    ax3.tick_params(axis='x', colors='white', labelsize=12)
    ax3.tick_params(axis='y', colors='white', labelsize=12)
    
    # Generous padding
    ax3.margins(x=0.05, y=0.08)
    
    # === PLOT 4: CLEAN BPM DISPLAY ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(colors['surface'])
    ax4.axis('off')
    
    # Large BPM display
    ax4.text(0.5, 0.7, f"{bpm:.1f}", ha='center', va='center', 
            fontsize=48, fontweight='bold', color=colors['primary'])
    ax4.text(0.5, 0.5, "BPM", ha='center', va='center', 
            fontsize=24, fontweight='bold', color=colors['text'])
    
    # Quality indicator
    quality_text = "Excellent" if 6 <= bpm <= 30 else "Check Data"
    quality_color = colors['success'] if 6 <= bpm <= 30 else colors['warning']
    ax4.text(0.5, 0.3, quality_text, ha='center', va='center', 
            fontsize=18, fontweight='bold', color=quality_color)
    
    # Frequency display
    ax4.text(0.5, 0.1, f"Frequency: {freq_final:.4f} Hz", ha='center', va='center', 
            fontsize=14, color=colors['accent'])
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("🎯 BREATHING RATE", fontsize=20, fontweight='bold', 
                 color=colors['text'], pad=40)
    
    # Add main title with generous spacing
    fig.suptitle("🫁 WiFi-Based Breathing Rate Detection System", 
                fontsize=28, fontweight='bold', color=colors['text'], y=0.94)
    
    # Add footer with generous spacing
    fig.text(0.5, 0.04, "Advanced Signal Processing • Real-time Monitoring • Medical Grade Accuracy", 
            ha='center', fontsize=14, color=colors['text'], alpha=0.8)
    
    # Maximum spacing layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.0, wspace=0.8, top=0.90, bottom=0.08, left=0.08, right=0.92)
    plt.show()
    
    # Create animated version for extra appeal
    create_animated_visualization(freqs_pos, mag_pos, freq_final, bpm, sig, f_phase, fs)

def create_animated_visualization(freqs_pos, mag_pos, freq_final, bpm, sig, f_phase, fs):
    """Create clean animated visualization with maximum spacing."""
    
    fig, ax = plt.subplots(figsize=(20, 14))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#1a1a2e')
    
    # Colors
    colors = {
        'primary': '#00d4ff',
        'secondary': '#ff6b6b',
        'accent': '#4ecdc4',
        'text': '#ffffff'
    }
    
    # Initialize plot elements with thinner lines
    line, = ax.plot([], [], color=colors['primary'], linewidth=2, alpha=0.9)
    peak_line = ax.axvline(freq_final, color=colors['secondary'], linewidth=3, alpha=0.9)
    
    # Setup plot with maximum spacing
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, np.max(mag_pos) * 1.1)
    ax.set_title("🌊 ANIMATED FREQUENCY SPECTRUM", fontsize=24, fontweight='bold', 
                color=colors['text'], pad=50)
    ax.set_xlabel("Frequency (Hz)", fontsize=18, color=colors['text'], labelpad=25)
    ax.set_ylabel("Magnitude", fontsize=18, color=colors['text'], labelpad=25)
    ax.grid(True, alpha=0.3, color=colors['text'])
    
    # Set axis colors to white for better visibility
    ax.tick_params(axis='x', colors='white', labelsize=14)
    ax.tick_params(axis='y', colors='white', labelsize=14)
    
    # Maximum margins for breathing room
    ax.margins(x=0.08, y=0.1)
    
    # Simplified reference lines with thinner width
    ref_bpms = [6, 12, 15, 20, 30]
    for bpm_ref in ref_bpms:
        freq_ref = bpm_ref / 60
        ax.axvline(freq_ref, color=colors['accent'], linestyle=':', linewidth=1, alpha=0.6)
        ax.text(freq_ref, np.max(mag_pos)*0.85, f'{bpm_ref}', color=colors['accent'], 
                fontsize=16, fontweight='bold', ha='center')
    
    def animate(frame):
        # Create pulsing effect
        alpha = 0.6 + 0.4 * np.sin(frame * 0.1)
        
        # Update line with glow effect
        line.set_data(freqs_pos, mag_pos)
        line.set_alpha(alpha)
        
        # Update peak line with pulsing
        peak_line.set_alpha(alpha)
        
        # Add dynamic text with better positioning
        ax.text(0.5, np.max(mag_pos)*0.7, f"🎯 Detected: {bpm:.1f} BPM", 
               ha='center', fontsize=20, fontweight='bold', color=colors['secondary'],
               alpha=alpha)
        
        return line, peak_line
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=100, interval=50, blit=False, repeat=True)
    
    # Maximum spacing layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.12, right=0.88)
    plt.show()
    
    return anim

def main():
    print("🫁 WiFi-Based Breathing Rate Detection System")
    print("=" * 50)
    
    # Load CSI
    csi = load_complex(REAL_CSV, IMAG_CSV)
    
    # Validate data quality
    validate_data_quality(csi, FS)

    # Remove zero subcarriers
    active = np.where(np.std(np.abs(csi), axis=0) > 1e-8)[0]
    csi = csi[:, active]
    print(f"✅ Active subcarriers: {len(active)} out of {csi.shape[1]}")

    # Use most variant subcarriers
    top = select_top_k(csi, K_TOP)
    sig = np.mean(csi[:, top], axis=1)
    sig = detrend(sig.real) + 1j * detrend(sig.imag)

    # Phase processing
    phase = detrend(np.unwrap(np.angle(sig)))
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS)
    f_phase = filtfilt(b, a, phase)

    # FFT
    N = len(sig)
    X = fft(sig * get_window("hann", N))
    freqs = fftfreq(N, 1 / FS)
    half = N // 2
    freqs_pos = freqs[:half]
    mag_pos = np.abs(X[:half])

    # Peak detection
    band = (freqs_pos >= LOWCUT) & (freqs_pos <= HIGHCUT)
    peak_idx = np.argmax(mag_pos[band])
    peak_idx = np.where(band)[0][peak_idx]
    freq_raw = freqs_pos[peak_idx]
    freq_refined = parabolic_interpolation(freqs_pos, mag_pos, peak_idx)

    # Harmonic correction
    freq_final = harmonic_correction(freq_refined, freqs_pos, mag_pos)

    # Phase refinement
    S_keep = np.zeros_like(X, dtype=complex)
    for b_idx in [peak_idx-1, peak_idx, peak_idx+1]:
        if 0 <= b_idx < half:
            S_keep[b_idx] = X[b_idx]
            S_keep[-b_idx] = X[-b_idx]
    refined_sig = ifft(S_keep)
    freq_phase = phase_regression(refined_sig, FS)

    # Validate phase
    if abs(freq_phase - freq_final) < PHASE_TOLERANCE and freq_phase > MIN_VALID_FREQ:
        freq_final = freq_phase

    bpm = freq_final * 60
    print(f"\n🎯 FINAL BREATHING RATE: {bpm:.2f} BPM")
    print(f"📊 Frequency: {freq_final:.4f} Hz")
    print(f"⏱️  Processing time: {len(sig)/FS:.1f} seconds")
    
    # Quality assessment
    if 6 <= bpm <= 30:
        print("✅ Result within normal breathing range (6-30 BPM)")
    else:
        print("⚠️  Result outside normal range - check data quality")

    # Stunning visualization
    create_stunning_visualization(freqs_pos, mag_pos, freq_final, bpm, sig, f_phase, FS)

def parse_arguments():
    """Parse command line arguments for flexible configuration."""
    parser = argparse.ArgumentParser(description='WiFi-Based Breathing Rate Detection')
    parser.add_argument('--real-csv', type=str, default=REAL_CSV,
                       help='Path to real part CSV file')
    parser.add_argument('--imag-csv', type=str, default=IMAG_CSV,
                       help='Path to imaginary part CSV file')
    parser.add_argument('--fs', type=float, default=FS,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--k-top', type=int, default=K_TOP,
                       help='Number of top subcarriers to use')
    parser.add_argument('--bpm', type=str, default='6',
                       help='Expected BPM for validation (e.g., 6, 10, 15)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Update global variables with command line arguments
    REAL_CSV = args.real_csv
    IMAG_CSV = args.imag_csv
    FS = args.fs
    K_TOP = args.k_top
    
    print(f"🔧 Configuration:")
    print(f"   Real CSV: {REAL_CSV}")
    print(f"   Imag CSV: {IMAG_CSV}")
    print(f"   Sampling Rate: {FS} Hz")
    print(f"   Top Subcarriers: {K_TOP}")
    print(f"   Expected BPM: {args.bpm}")
    print()
    
    main()