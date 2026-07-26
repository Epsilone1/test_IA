import importlib.metadata
import subprocess
import sys
import os
import json

from pathlib import Path

Version_torch_GPU = "cu130"

data = {}
Version_torch_GPU_User = "None"
CPU_GPU_mode = "GPU"
TempJSON = Path(__file__).parent / "Temp.json"
try:
    data = json.loads(TempJSON.read_text(encoding="utf-8"))
except:
    pass
data["Version_torch_GPU_User"] = data.get("Version_torch_GPU_User",Version_torch_GPU_User)
data["CPU_GPU_mode"] = data.get("CPU_GPU_mode",CPU_GPU_mode)
TempJSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_torch_installed():
    torch_present = False
    try:
        importlib.metadata.version("torch")
        torch_present = True
    except importlib.metadata.PackageNotFoundError:
        torch_present = False
    return torch_present

def rapport():
    print(f"Version_torch_GPU_User: {data['Version_torch_GPU_User']}")
    TempJSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

def Version_Control():

    if data["Version_torch_GPU_User"] == "suppression":
        if is_torch_installed():
            import torch
            if torch.cuda.is_available():
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "-y", "torchvision"])
        rapport()
        return

    
    if data["Version_torch_GPU_User"] == Version_torch_GPU or data["Version_torch_GPU_User"] == "Non_Compatible":
        rapport()
        return
    if data["CPU_GPU_mode"] == "CPU":
        rapport()
        return

    
    if not is_torch_installed():
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        if CPU_GPU_mode == "CPU":
            rapport()
            return
    import torch
    if  (not torch.cuda.is_available() and "--deja-tente" not in sys.argv) or (not torch.__version__[-5:] == Version_torch_GPU and "--deja-tente" not in sys.argv):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "-y", "torchvision"])
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/" + Version_torch_GPU])
        os.execv(sys.executable, [sys.executable] + sys.argv + ["--deja-tente"])
    if is_torch_installed():
        import torch
        data["Version_torch_GPU_User"] = torch.__version__[-5:]
    else:
        data["Version_torch_GPU_User"] = "Non_Compatible"
    rapport()

    

