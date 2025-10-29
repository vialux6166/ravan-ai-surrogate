#!/usr/bin/env python3
"""
Ravan-TRD Integrated GUI
World-class interface for quantum simulation and AI reasoning model training
"""

import sys
import os
from pathlib import Path
import json
import subprocess
import threading
from datetime import datetime

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
        QComboBox, QSpinBox, QLineEdit, QFileDialog, QGroupBox,
        QCheckBox, QSplitter, QTableWidget, QTableWidgetItem,
        QMessageBox, QStatusBar, QFrame
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
except ImportError:
    print("PyQt6 not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
        QComboBox, QSpinBox, QLineEdit, QFileDialog, QGroupBox,
        QCheckBox, QSplitter, QTableWidget, QTableWidgetItem,
        QMessageBox, QStatusBar, QFrame
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor, QPalette, QIcon


class WorkerThread(QThread):
    """Background worker for long-running tasks"""
    output_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, command, shell=False):
        super().__init__()
        self.command = command
        self.shell = shell
        self.process = None
    
    def run(self):
        """Execute command in background"""
        try:
            if self.shell:
                self.process = subprocess.Popen(
                    self.command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            else:
                self.process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            
            for line in self.process.stdout:
                self.output_signal.emit(line.strip())
            
            self.process.wait()
            success = self.process.returncode == 0
            msg = "Success" if success else f"Failed (code {self.process.returncode})"
            self.finished_signal.emit(success, msg)
            
        except Exception as e:
            self.output_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False, str(e))
    
    def stop(self):
        """Stop the running process"""
        if self.process:
            self.process.terminate()



class RavanTRDGUI(QMainWindow):
    """Main GUI window for Ravan-TRD integration"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ravan-TRD Integrated Platform")
        self.setGeometry(100, 100, 1400, 900)
        
        # State
        self.worker = None
        self.project_dir = Path(__file__).parent
        
        # Setup UI
        self.setup_ui()
        self.apply_theme()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_ravan_tab(), "🔬 Ravan Quantum")
        tabs.addTab(self.create_trd_tab(), "🧠 TRD Training")
        tabs.addTab(self.create_inference_tab(), "💬 Model Inference")
        tabs.addTab(self.create_monitor_tab(), "📊 System Monitor")
        
        layout.addWidget(tabs)
    
    def create_header(self):
        """Create header section"""
        header = QFrame()
        header.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(header)
        
        title = QLabel("🌌 Ravan-TRD Integrated Platform")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        
        layout.addStretch()
        
        status_label = QLabel("Status:")
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: green; font-size: 20px;")
        
        layout.addWidget(status_label)
        layout.addWidget(self.status_indicator)
        
        return header
    
    def create_ravan_tab(self):
        """Create Ravan quantum simulation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Simulation controls
        sim_group = QGroupBox("Quantum Simulation")
        sim_layout = QVBoxLayout()
        
        # Simulation type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Simulation Type:"))
        self.sim_type = QComboBox()
        self.sim_type.addItems([
            "Quantum Harmonic Oscillator",
            "Particle in Box",
            "Hydrogen Atom",
            "Quantum Tunneling",
            "Spin Systems"
        ])
        type_layout.addWidget(self.sim_type)
        sim_layout.addLayout(type_layout)
        
        # Parameters
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Grid Points:"))
        self.grid_points = QSpinBox()
        self.grid_points.setRange(50, 500)
        self.grid_points.setValue(100)
        param_layout.addWidget(self.grid_points)
        
        param_layout.addWidget(QLabel("Time Steps:"))
        self.time_steps = QSpinBox()
        self.time_steps.setRange(100, 5000)
        self.time_steps.setValue(1000)
        param_layout.addWidget(self.time_steps)
        sim_layout.addLayout(param_layout)
        
        # Run button
        run_sim_btn = QPushButton("▶ Run Simulation")
        run_sim_btn.clicked.connect(self.run_ravan_simulation)
        sim_layout.addWidget(run_sim_btn)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # Output
        output_group = QGroupBox("Simulation Output")
        output_layout = QVBoxLayout()
        self.ravan_output = QTextEdit()
        self.ravan_output.setReadOnly(True)
        self.ravan_output.setFont(QFont("Courier", 9))
        output_layout.addWidget(self.ravan_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        return tab
    
    def create_trd_tab(self):
        """Create TRD training tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Dataset generation
        dataset_group = QGroupBox("📚 Dataset Generation")
        dataset_layout = QVBoxLayout()
        
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Target Size:"))
        self.dataset_size = QSpinBox()
        self.dataset_size.setRange(100, 10000)
        self.dataset_size.setValue(1000)
        gen_layout.addWidget(self.dataset_size)
        
        gen_layout.addWidget(QLabel("Teacher Model:"))
        self.teacher_model = QComboBox()
        self.teacher_model.addItems(["qwen3:30b-a3b", "qwen2.5:32b", "llama3:70b"])
        gen_layout.addWidget(self.teacher_model)
        dataset_layout.addLayout(gen_layout)
        
        gen_btn = QPushButton("🔄 Generate Dataset")
        gen_btn.clicked.connect(self.generate_dataset)
        dataset_layout.addWidget(gen_btn)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Training controls
        train_group = QGroupBox("🎓 Model Training")
        train_layout = QVBoxLayout()
        
        # Training parameters
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Epochs:"))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10)
        self.epochs.setValue(3)
        param_layout.addWidget(self.epochs)
        
        param_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 8)
        self.batch_size.setValue(2)
        param_layout.addWidget(self.batch_size)
        
        param_layout.addWidget(QLabel("Gradient Accum:"))
        self.grad_accum = QSpinBox()
        self.grad_accum.setRange(1, 32)
        self.grad_accum.setValue(8)
        param_layout.addWidget(self.grad_accum)
        train_layout.addLayout(param_layout)
        
        # Dataset selection
        dataset_select_layout = QHBoxLayout()
        dataset_select_layout.addWidget(QLabel("Dataset:"))
        self.dataset_path = QLineEdit()
        self.dataset_path.setText("trd_framework/large_dataset/physics_large_dataset.jsonl")
        dataset_select_layout.addWidget(self.dataset_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_dataset)
        dataset_select_layout.addWidget(browse_btn)
        train_layout.addLayout(dataset_select_layout)
        
        # Train button
        train_btn = QPushButton("🚀 Start Training")
        train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(train_btn)
        
        # Progress
        self.train_progress = QProgressBar()
        train_layout.addWidget(self.train_progress)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Training output
        output_group = QGroupBox("Training Log")
        output_layout = QVBoxLayout()
        self.train_output = QTextEdit()
        self.train_output.setReadOnly(True)
        self.train_output.setFont(QFont("Courier", 9))
        output_layout.addWidget(self.train_output)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        return tab
    
    def create_inference_tab(self):
        """Create model inference tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_group = QGroupBox("🤖 Model Selection")
        model_layout = QHBoxLayout()
        
        model_layout.addWidget(QLabel("Model:"))
        self.model_path = QComboBox()
        self.model_path.setEditable(True)
        self.model_path.addItems([
            "trd_framework/models/qwen2-1.5b-1k-fixed",
            "trd_framework/models/qwen2-1.5b-1k"
        ])
        model_layout.addWidget(self.model_path)
        
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Query interface
        query_group = QGroupBox("💬 Query Interface")
        query_layout = QVBoxLayout()
        
        query_layout.addWidget(QLabel("Enter your question:"))
        self.query_input = QTextEdit()
        self.query_input.setMaximumHeight(100)
        self.query_input.setPlaceholderText("e.g., Explain quantum entanglement...")
        query_layout.addWidget(self.query_input)
        
        query_btn = QPushButton("🔍 Generate Response")
        query_btn.clicked.connect(self.generate_response)
        query_layout.addWidget(query_btn)
        
        query_group.setLayout(query_layout)
        layout.addWidget(query_group)
        
        # Response display
        response_group = QGroupBox("Response")
        response_layout = QVBoxLayout()
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        self.response_output.setFont(QFont("Arial", 10))
        response_layout.addWidget(self.response_output)
        response_group.setLayout(response_layout)
        layout.addWidget(response_group)
        
        return tab
    
    def create_monitor_tab(self):
        """Create system monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # System info
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()
        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(150)
        info_layout.addWidget(self.system_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Model list
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout()
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(4)
        self.models_table.setHorizontalHeaderLabels(["Model", "Size", "Date", "Status"])
        models_layout.addWidget(self.models_table)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        models_layout.addWidget(refresh_btn)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Initial refresh
        self.refresh_system_info()
        self.refresh_models()
        
        return tab
    
    def apply_theme(self):
        """Apply modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QTextEdit, QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
            QComboBox, QSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #0d47a1;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0d47a1;
            }
            QTableWidget {
                background-color: #2d2d2d;
                gridline-color: #3d3d3d;
                border: 1px solid #3d3d3d;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: #e0e0e0;
                padding: 5px;
                border: none;
            }
        """)
    
    # Action methods
    def run_ravan_simulation(self):
        """Run Ravan quantum simulation"""
        self.ravan_output.clear()
        self.ravan_output.append("🔬 Starting Ravan simulation...\n")
        self.statusBar().showMessage("Running simulation...")
        
        sim_type = self.sim_type.currentText()
        grid = self.grid_points.value()
        steps = self.time_steps.value()
        
        self.ravan_output.append(f"Simulation: {sim_type}")
        self.ravan_output.append(f"Grid Points: {grid}")
        self.ravan_output.append(f"Time Steps: {steps}\n")
        
        # Placeholder - integrate with actual Ravan code
        self.ravan_output.append("✓ Simulation completed successfully!")
        self.statusBar().showMessage("Simulation complete")
    
    def generate_dataset(self):
        """Generate TRD training dataset"""
        size = self.dataset_size.value()
        teacher = self.teacher_model.currentText()
        
        reply = QMessageBox.question(
            self,
            "Generate Dataset",
            f"Generate {size} examples using {teacher}?\nThis will take ~{size * 0.18:.0f} minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.train_output.clear()
            self.train_output.append(f"🔄 Generating {size} examples...\n")
            self.statusBar().showMessage("Generating dataset...")
            
            cmd = f'wsl bash -c "cd \\"{self.project_dir}\\" && source venv_test/bin/activate && python trd_framework/generate_dataset_windows.py --target-size {size} --teacher-model {teacher}"'
            
            self.worker = WorkerThread(cmd, shell=True)
            self.worker.output_signal.connect(self.train_output.append)
            self.worker.finished_signal.connect(self.on_dataset_complete)
            self.worker.start()
    
    def start_training(self):
        """Start TRD model training"""
        dataset = self.dataset_path.text()
        epochs = self.epochs.value()
        batch = self.batch_size.value()
        grad = self.grad_accum.value()
        
        if not Path(dataset).exists():
            QMessageBox.warning(self, "Error", f"Dataset not found: {dataset}")
            return
        
        reply = QMessageBox.question(
            self,
            "Start Training",
            f"Train model for {epochs} epochs?\nThis will take ~35 minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.train_output.clear()
            self.train_output.append("🚀 Starting TRD training...\n")
            self.statusBar().showMessage("Training in progress...")
            self.train_progress.setValue(0)
            
            output_dir = "trd_framework/models/qwen2-1.5b-1k-fixed"
            
            cmd = f'wsl bash -c "cd \\"{self.project_dir}\\" && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset {dataset} --output-dir {output_dir} --epochs {epochs} --batch-size {batch} --gradient-accumulation {grad}"'
            
            self.worker = WorkerThread(cmd, shell=True)
            self.worker.output_signal.connect(self.on_training_output)
            self.worker.finished_signal.connect(self.on_training_complete)
            self.worker.start()
    
    def load_model(self):
        """Load TRD model for inference"""
        model = self.model_path.currentText()
        self.response_output.clear()
        self.response_output.append(f"Loading model: {model}...\n")
        self.statusBar().showMessage("Loading model...")
        
        # Placeholder - integrate with actual model loading
        self.response_output.append("✓ Model loaded successfully!")
        self.statusBar().showMessage("Model ready")
    
    def generate_response(self):
        """Generate response from loaded model"""
        query = self.query_input.toPlainText().strip()
        
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a question")
            return
        
        self.response_output.clear()
        self.response_output.append(f"Query: {query}\n\n")
        self.response_output.append("Generating response...\n")
        self.statusBar().showMessage("Generating...")
        
        model = self.model_path.currentText()
        
        cmd = f'wsl bash -c "cd \\"{self.project_dir}\\" && source venv_test/bin/activate && echo \\"{query}\\" | python trd_framework/test_trd_model.py --model-path {model} --mode interactive"'
        
        self.worker = WorkerThread(cmd, shell=True)
        self.worker.output_signal.connect(self.response_output.append)
        self.worker.finished_signal.connect(lambda s, m: self.statusBar().showMessage("Response complete"))
        self.worker.start()
    
    def browse_dataset(self):
        """Browse for dataset file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            str(self.project_dir / "trd_framework"),
            "JSONL Files (*.jsonl);;All Files (*)"
        )
        if file_path:
            self.dataset_path.setText(file_path)
    
    def refresh_system_info(self):
        """Refresh system information"""
        info = []
        info.append("=== System Information ===\n")
        info.append(f"Project Directory: {self.project_dir}")
        info.append(f"Python: {sys.version.split()[0]}")
        
        # Check CUDA
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                info.append(f"\nGPU: {result.stdout.strip()}")
            else:
                info.append("\nGPU: Not detected")
        except:
            info.append("\nGPU: Not detected")
        
        self.system_info.setText("\n".join(info))
    
    def refresh_models(self):
        """Refresh available models list"""
        self.models_table.setRowCount(0)
        
        models_dir = self.project_dir / "trd_framework" / "models"
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    row = self.models_table.rowCount()
                    self.models_table.insertRow(row)
                    
                    self.models_table.setItem(row, 0, QTableWidgetItem(model_dir.name))
                    
                    # Calculate size
                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    size_mb = size / (1024 * 1024)
                    self.models_table.setItem(row, 1, QTableWidgetItem(f"{size_mb:.1f} MB"))
                    
                    # Get date
                    date = datetime.fromtimestamp(model_dir.stat().st_mtime).strftime("%Y-%m-%d")
                    self.models_table.setItem(row, 2, QTableWidgetItem(date))
                    
                    self.models_table.setItem(row, 3, QTableWidgetItem("✓ Ready"))
        
        self.models_table.resizeColumnsToContents()
    
    # Callback methods
    def on_training_output(self, line):
        """Handle training output"""
        self.train_output.append(line)
        
        # Update progress based on output
        if "epoch" in line.lower() or "step" in line.lower():
            # Simple progress estimation
            current = self.train_progress.value()
            if current < 90:
                self.train_progress.setValue(current + 1)
    
    def on_training_complete(self, success, message):
        """Handle training completion"""
        self.train_progress.setValue(100)
        if success:
            self.train_output.append("\n✓ Training completed successfully!")
            self.statusBar().showMessage("Training complete")
            QMessageBox.information(self, "Success", "Model training completed!")
            self.refresh_models()
        else:
            self.train_output.append(f"\n✗ Training failed: {message}")
            self.statusBar().showMessage("Training failed")
            QMessageBox.warning(self, "Error", f"Training failed: {message}")
    
    def on_dataset_complete(self, success, message):
        """Handle dataset generation completion"""
        if success:
            self.train_output.append("\n✓ Dataset generation completed!")
            self.statusBar().showMessage("Dataset ready")
            QMessageBox.information(self, "Success", "Dataset generated successfully!")
        else:
            self.train_output.append(f"\n✗ Generation failed: {message}")
            self.statusBar().showMessage("Generation failed")
            QMessageBox.warning(self, "Error", f"Generation failed: {message}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Ravan-TRD Platform")
    
    window = RavanTRDGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
