# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Uninstall current CPU-only PyTorch
pip uninstall -y torch torchaudio

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.9)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'None')"
