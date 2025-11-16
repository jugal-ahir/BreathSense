# WiFi-Based Breathing Rate Detection System

A Python implementation for detecting breathing rates using WiFi Channel State Information (CSI) data.

## Features

- ✅ **Harmonic Correction**: Prevents false detection of harmonics (e.g., 10 BPM → 34 BPM)
- ✅ **Bandpass Filtering**: Focuses on 3-40 BPM range (0.05-0.7 Hz)
- ✅ **Phase Regression**: Advanced phase analysis with safety checks
- ✅ **Enhanced Visualization**: Comprehensive 4-panel plots
- ✅ **Data Validation**: Quality checks and error handling
- ✅ **Command Line Interface**: Flexible configuration options

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python main.py --help
   ```

## Usage

### Basic Usage
```bash
python main.py
```

### Advanced Usage with Custom Parameters
```bash
python main.py --real-csv "path/to/real.csv" --imag-csv "path/to/imag.csv" --fs 10.0 --k-top 50
```

### Command Line Options
- `--real-csv`: Path to real part CSV file
- `--imag-csv`: Path to imaginary part CSV file  
- `--fs`: Sampling frequency in Hz (default: 10.0)
- `--k-top`: Number of top subcarriers to use (default: 50)
- `--bpm`: Expected BPM for validation (e.g., 6, 10, 15)

## Data Format

The system expects two CSV files:
- **Real part**: `config0001_csi_real_log.csv`
- **Imaginary part**: `config0001_csi_imag_log.csv`

Each file should contain CSI data with:
- No header row
- Each row represents a time sample
- Each column represents a subcarrier

## Output

The system provides:
1. **Console output** with processing details and final BPM
2. **4-panel visualization**:
   - Frequency spectrum with detected peak
   - Time domain signal (real/imaginary parts)
   - Filtered phase signal
   - BPM range visualization

## Example Output

```
🫁 WiFi-Based Breathing Rate Detection System
==================================================
✅ Loaded CSI data: 1000 samples, 64 subcarriers
📊 Data duration: 100.0 seconds
✅ Active subcarriers: 60 out of 64
✅ Selected top 50 subcarriers out of 60

🎯 FINAL BREATHING RATE: 12.34 BPM
📊 Frequency: 0.2057 Hz
⏱️  Processing time: 100.0 seconds
✅ Result within normal breathing range (6-30 BPM)
```

## Troubleshooting

### Common Issues

1. **File not found error**: Check that CSV file paths are correct
2. **Empty data warning**: Ensure CSV files contain valid data
3. **Short duration warning**: Use at least 10 seconds of data for accuracy
4. **Out of range BPM**: Check data quality and environment conditions

### Data Quality Tips

- Use at least 10-30 seconds of data
- Ensure stable WiFi connection during data collection
- Minimize movement and interference
- Use data from multiple subcarriers for better accuracy

## Technical Details

- **Frequency Range**: 3-40 BPM (0.05-0.67 Hz)
- **Filtering**: 4th order Butterworth bandpass filter
- **Window Function**: Hann window for FFT
- **Peak Detection**: Parabolic interpolation for sub-bin accuracy
- **Harmonic Detection**: Checks for /2 and /3 harmonics

## License

This project is for educational and research purposes.
