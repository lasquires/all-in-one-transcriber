Here’s a drop-in **README.md** you can ship with your project.

---

# Transcribe App — Setup & Run (Windows, NVIDIA GPU)

This guide verifies you’re on an **NVIDIA GPU** machine and then initializes the Conda environment from `environment.yml` (named `transcribe`). An optional block shows how to install Miniconda via PowerShell if Conda isn’t installed yet.

---
## **Pre-requisites**
### Verify NVIDIA GPU + driver

Open **PowerShell** and run:

```powershell
# List graphics adapters
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion

# If NVIDIA is present, this should work and show your GPU + driver
nvidia-smi
```

**You need:**

* An **NVIDIA GPU** listed above
* A recent **NVIDIA driver** (for CUDA 12.1 builds, driver ≥ **531.xx** is recommended)

If `nvidia-smi` isn’t found, install/update the NVIDIA driver from GeForce/Studio drivers (or your IT-managed image).

---

### Install Miniconda via PowerShell --- ***Skip if conda already installed***

If `conda` isn’t recognized:

```powershell
# Install Miniconda (64-bit) using winget
winget install -e --id Anaconda.Miniconda3

# Open a new PowerShell window (important), then initialize conda for PS:
conda init powershell

# Close and re-open PowerShell so your profile changes take effect
```

> If you cannot use winget, download Miniconda from the official site and run the installer. After installing, run `conda init powershell`, then restart PowerShell.

---

## **Setup**
### Create the environment from `environment.yml`

From the project folder (the one containing `environment.yml`):

```powershell
# Create the env (name is defined inside environment.yml) 
# (this takes about 5 minutes)
conda env create -f .\environment.yml


# (Optional) Verify torch build & CUDA availability
python -c "import torch; print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

You should see `CUDA available: True` on a working NVIDIA machine.

---

## **Run**

From the project folder find and click `run_transcribe.bat` 

You can then make a shortcut of this file and place the shortcut on desktop or wherever.

