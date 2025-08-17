<div align="center">
<img src="https://placehold.co/1200x300/007BFF/FFFFFF?text=FEEL-ECG+Implementation&font=lato" alt="FEEL-ECG Banner">
</div>

<h1 align="center">FEEL-ECG: A Python Implementation</h1>

<div align="center">
<a href="https://www.python.org/downloads/release/python-380/">
<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python 3.8+">
</a>
<a href="https://github.com/tensorflow/tensorflow">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow 2.x">
</a>
<a href="https://opensource.org/licenses/MIT">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT">
</a>
</div>

This repository contains a complete Python implementation of the paper "FEEL-ECG: Federated Edge Learning for Personalized and Explainable ECG Monitoring with Adaptive Compression and Preprocessing". It reproduces the core framework for building a lightweight, private, and interpretable model for real-time arrhythmia detection on edge devices.

📋 Table of Contents
📊 Results Panel

✨ Key Features

🚀 Getting Started

⚙️ How to Run the Pipeline

📁 Project Structure

🤝 Contributing

📜 License

📊 Results Panel
This implementation is designed to achieve results that closely align with the performance metrics reported in the original paper. The table below outlines the target goals based on the paper's findings.

Metric

Paper's Result

🎯 Implementation Goal

Diagnostic Accuracy

98.2%

~98%

Final Model Size

1.3 MB

~1.3 MB

Inference Latency

8 ms (on RPi 4)

< 10 ms

Power Consumption

1.8 W (Duty-Cycled)

N/A (Simulated)

XAI Footprint

112 KB

~112 KB (Simulated)

✨ Key Features
🔬 Data Preprocessing: Automated download and processing of the MIT-BIH Arrhythmia Database using R-peak detection and AAMI standard labeling.

✂️ Hybrid Compression: A three-stage pipeline including architectural compression, Quantization-Aware Training (QAT), and structured pruning.

🌐 Federated Learning: A simulation of Motif-Aware Federated Averaging (MA-FedAvg) with differential privacy to protect user data.

💡 Edge-Ready Model: Generates a final, highly compressed .tflite model ready for deployment on resource-constrained devices like Raspberry Pi.

⚡ On-Device Simulation: Includes a script to simulate inference, explainability, and energy-efficient duty cycling on an edge device.

🚀 Getting Started
Follow these steps to set up the project on your local machine.

Prerequisites
Python 3.8 or higher

Git

Installation and Setup
Clone the repository:

git clone https://github.com/your-username/FEEL-ECG-Implementation.git
cd FEEL-ECG-Implementation

Create a virtual environment and install dependencies:
This isolates the project's dependencies from your system's Python installation.

# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt

⚙️ How to Run the Pipeline
Execute the scripts in the following order to run the complete pipeline from data preparation to model generation.

1. Preprocess the Data
This script will download the MIT-BIH dataset into a data/ directory (which is ignored by Git) and generate ecg_segments.npy and ecg_labels.npy.

python preprocess_data.py

2. Create and Compress the Centralized Model
This script trains the model using the three-stage compression pipeline and saves the final feel_ecg_model.tflite file.

python create_compressed_model.py

3. Run the Federated Learning Simulation
This script simulates the MA-FedAvg process with 10 clients and saves the final aggregated model as feel_ecg_federated_model.h5.

python federated_simulation.py

4. Simulate On-Device Inference
This script uses the .tflite model to simulate real-time, QRS-triggered inference. It can be run on your development machine or a target Raspberry Pi.

python run_on_pi.py

📁 Project Structure
The repository is organized as follows:

.
├── .gitignore                 # Files and folders to be ignored by Git
├── README.md                  # This file
├── requirements.txt           # Project dependencies
├── preprocess_data.py         # Handles data download and preparation
├── create_compressed_model.py # Builds and compresses the centralized model
├── federated_simulation.py    # Simulates the MA-FedAvg process
└── run_on_pi.py               # Simulates on-device inference and XAI

🤝 Contributing
Contributions are welcome! If you have suggestions for improving this project, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
