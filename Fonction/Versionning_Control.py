import importlib.metadata
import subprocess
import sys
import os

Version_torch_GPU = "cu130"

def Version_Control():
    try:
        importlib.metadata.version("torch")
        torch_present = True
    except importlib.metadata.PackageNotFoundError:
        torch_present = False

    if torch_present:
        import torch
            
        if  (not torch.cuda.is_available() and "--deja-tente" not in sys.argv) or (not torch.__version__[-5:] == Version_torch_GPU and "--deja-tente" not in sys.argv):
            print("torch Nok")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "-y", "torchvision"])
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/" + Version_torch_GPU])
            os.execv(sys.executable, [sys.executable] + sys.argv + ["--deja-tente"])
        else:
            print("GPU ok")

Version_Control()