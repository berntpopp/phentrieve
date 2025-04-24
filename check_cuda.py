import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # Test CUDA with a simple operation
    x = torch.rand(5, 5).cuda()
    print(f"Tensor device: {x.device}")
    print("CUDA is working correctly!")
else:
    print("CUDA is not available. Check your installation.")
