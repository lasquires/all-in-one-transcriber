Here’s a drop-in **README.md** you can ship with your project.

---

# Transcribe App — Setup & Run (Windows, NVIDIA GPU)

This guide verifies you’re on an **NVIDIA GPU** machine and then initializes the Conda environment from `environment.yml` (named `transcribe`). An optional block shows how to install Miniconda via PowerShell if Conda isn’t installed yet.

---

## 0) Verify NVIDIA GPU + driver

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

## 1) (Optional) Install Miniconda via PowerShell

If `conda` isn’t recognized:

```powershell
# Install Miniconda (64-bit) using winget
winget install -e --id Anaconda.Miniconda3

# Open a new PowerShell window (important), then initialize conda for PS:
conda init powershell

# Close and reopen PowerShell so your profile changes take effect
```

> If you cannot use winget, download Miniconda from the official site and run the installer. After installing, run `conda init powershell`, then restart PowerShell.

---

## 2) Create the environment from `environment.yml`

From the project folder (the one containing `environment.yml`):

```powershell
# Create the env (name is defined inside environment.yml)
conda env create -f .\environment.yml

# Activate it
conda activate transcribe

# (Optional) Verify torch build & CUDA availability
python -c "import torch; print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

You should see `CUDA available: True` on a working NVIDIA machine.

---

## 3) First run

From the project folder:

```powershell
conda activate transcribe
python .\transcribe.py
```

---

## 4) Handy one-shot batch launcher (optional)

Create a `run_transcribe.bat` next to `trancribe.py`:

```bat
@echo off
SET "CONDA_PATH=%USERPROFILE%\miniconda3"
CALL "%CONDA_PATH%\Scripts\activate.bat"
CALL conda activate transcribe
CD /d %~dp0
python transcribe.py
pause
```

> If Conda is installed somewhere else (e.g., `C:\ProgramData\miniconda3` or `C:\Users\<you>\anaconda3`), update `CONDA_PATH`.

---

## 5) Troubleshooting (quick)

* **`nvidia-smi` not found** → Install/repair NVIDIA driver; ensure it’s an NVIDIA GPU.
* **`CUDA available: False`** → Update NVIDIA driver (≥ 531.xx), reboot, recheck.


  *(Add any extras you need; if you add a new dependency, re-export or update `environment.yml` for teammates.)*
* **Conda not recognized** → Open a new PowerShell window after `conda init powershell`, or call Conda explicitly:

  ```powershell
  & "$env:USERPROFILE\miniconda3\condabin\conda.bat" env list
  ```

---

## 6) Re-creating or updating the env

When `environment.yml` changes:

```powershell
# Recreate cleanly (removes the old env)
conda remove -n transcribe --all -y
conda env create -f .\environment.yml
conda activate transcribe
```

Or update in place (lighter, may not resolve major changes):

```powershell
conda env update -f .\environment.yml --prune
```

---

That’s it—once you see `CUDA available: True` and the app starts without import errors, you’re good to go.

