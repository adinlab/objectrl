# Troubleshooting & Common Issues

This document addresses some common issues users may encounter when installing and running ObjectRL.

---

## 1. Installing Extras with `pip` in `zsh` or PowerShell

When installing optional dependencies like the documentation extras, some shells (e.g., `zsh` or PowerShell) require quoting the package specifier:

```bash
# In bash:
pip install objectrl[docs]

# In zsh or PowerShell:
pip install "objectrl[docs]"
```

## 2. CUDA Device Errors

By default, ObjectRL attempts to use the CUDA device for GPU acceleration. If you run ObjectRL on a machine without CUDA support, you may encounter errors.

**Solution:** Force CPU usage with the flag:

```bash 
python -m objectrl.main --model.name sac --system.device "cpu"
```

## 3. Permission Denied When Saving Logs

The default log directory is set to `../_logs`, which may not be writable when installed via `pip`. This can cause a `PermissionDenied` error during training or evaluation.

**Solution:** Specify a writable log directory with:

```bash
python -m objectrl.main --model.name sac --logging.result_path ./_logs
```

---

If you have any other issues, please check the [GitHub Issues](https://github.com/adinlab/objectrl/issues) page or open a new issue.