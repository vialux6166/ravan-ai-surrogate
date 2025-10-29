# 🌌 Ravan-TRD Integrated Platform GUI

World-class graphical interface for quantum simulation and AI reasoning model training.

## Features

### 🔬 Ravan Quantum Simulation
- Interactive quantum simulation controls
- Multiple simulation types:
  - Quantum Harmonic Oscillator
  - Particle in Box
  - Hydrogen Atom
  - Quantum Tunneling
  - Spin Systems
- Configurable parameters (grid points, time steps)
- Real-time output visualization

### 🧠 TRD Training Pipeline
- **Dataset Generation**
  - Configurable dataset size (100-10,000 examples)
  - Multiple teacher models (qwen3:30b-a3b, qwen2.5:32b, llama3:70b)
  - Progress tracking with ETA
  
- **Model Training**
  - Adjustable hyperparameters (epochs, batch size, gradient accumulation)
  - Real-time training logs
  - Progress bar with status updates
  - Automatic EOS token handling (fixed version)

### 💬 Model Inference
- Load trained TRD models
- Interactive query interface
- Structured reasoning responses
- Real-time generation

### 📊 System Monitor
- System information display
- GPU detection and stats
- Available models list with sizes and dates
- One-click model refresh

## Installation

### Prerequisites
```bash
pip install PyQt6
```

The GUI will auto-install PyQt6 if not found.

## Usage

### Quick Start
Double-click: `LAUNCH_GUI.bat`

Or run manually:
```bash
python ravan_trd_gui.py
```

### Workflow

1. **Generate Dataset** (TRD Training tab)
   - Set target size (e.g., 1000)
   - Select teacher model
   - Click "Generate Dataset"
   - Wait ~3 hours for 1000 examples

2. **Train Model** (TRD Training tab)
   - Configure training parameters
   - Select dataset file
   - Click "Start Training"
   - Wait ~35 minutes

3. **Test Model** (Model Inference tab)
   - Select trained model
   - Click "Load Model"
   - Enter your question
   - Click "Generate Response"

4. **Monitor System** (System Monitor tab)
   - View system info
   - Check available models
   - Refresh to see new models

## Interface Overview

### Modern Dark Theme
- Professional dark color scheme
- High contrast for readability
- Smooth animations and transitions
- Responsive layout

### Tabs
1. **🔬 Ravan Quantum** - Quantum simulations
2. **🧠 TRD Training** - Dataset generation and model training
3. **💬 Model Inference** - Query trained models
4. **📊 System Monitor** - System info and model management

### Status Bar
- Real-time status updates
- Operation progress indicators
- Success/error notifications

## Features

### Background Processing
- All long-running tasks run in background threads
- UI remains responsive during operations
- Real-time output streaming
- Cancellable operations

### Error Handling
- Comprehensive error messages
- User-friendly dialogs
- Automatic validation
- Safe operation confirmations

### File Management
- Browse for datasets
- Auto-detect models
- Display file sizes and dates
- Quick access to common paths

## Technical Details

### Architecture
- **PyQt6** - Modern Qt6 bindings for Python
- **Threading** - Background workers for long operations
- **Subprocess** - WSL integration for training
- **Real-time Updates** - Signal/slot mechanism for UI updates

### Integration
- **Ravan** - Quantum simulation framework
- **TRD** - Topological Resonance Distillation
- **WSL** - Windows Subsystem for Linux for training
- **Ollama** - Teacher model API

### Performance
- Non-blocking UI operations
- Efficient subprocess management
- Minimal memory footprint
- GPU acceleration support

## Troubleshooting

### GUI doesn't start
```bash
pip install --upgrade PyQt6
```

### Training fails
- Check WSL is installed and running
- Verify venv_test environment exists
- Ensure dataset path is correct
- Check CUDA/GPU availability

### Model loading issues
- Verify model path exists
- Check model files are complete
- Ensure sufficient memory

## Advanced Usage

### Custom Themes
Edit the `apply_theme()` method in `ravan_trd_gui.py` to customize colors.

### Add New Simulations
Extend `create_ravan_tab()` to add new simulation types.

### Custom Models
Add model paths to the dropdown in `create_inference_tab()`.

## Screenshots

The GUI features:
- Clean, modern interface
- Intuitive tab-based navigation
- Real-time progress tracking
- Professional dark theme
- Responsive controls

## Support

For issues or questions:
1. Check the training logs in the GUI
2. Review error messages in dialogs
3. Verify system requirements
4. Check file paths and permissions

## Future Enhancements

- [ ] Real-time GPU monitoring
- [ ] Training metrics visualization
- [ ] Model comparison tools
- [ ] Export/import configurations
- [ ] Batch processing
- [ ] Remote training support
- [ ] Advanced visualization for Ravan simulations
- [ ] Model performance benchmarks

## License

Part of the Ravan-TRD integrated platform.
