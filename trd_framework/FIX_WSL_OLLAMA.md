# Fix: WSL Cannot Connect to Ollama

## Problem

WSL cannot connect to Ollama running on Windows:
```
Error: Connection refused to localhost:11434
```

## Solution: Use Windows Python Instead

### Option 1: Run in Windows (Easiest)

**Double-click**: `trd_framework\GENERATE_1K_WINDOWS.bat`

This runs Python in Windows (not WSL), avoiding the networking issue.

### Option 2: Manual Windows Command

```cmd
# Terminal 1: Start Ollama
ollama run qwen3:30b-a3b

# Terminal 2: Generate dataset (Windows Python)
python trd_framework\generate_dataset_windows.py --target-size 1000 --teacher-model qwen3:30b-a3b
```

### Option 3: Fix WSL Networking

If you want to use WSL, get the Windows host IP:

```bash
# In WSL, get Windows IP
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
# Example output: 172.24.176.1

# Use that IP
python trd_framework/generate_large_dataset.py \
    --target-size 1000 \
    --teacher-url http://172.24.176.1:11434
```

## Why This Happens

- Ollama runs on Windows
- WSL is a separate network namespace
- `localhost` in WSL ≠ `localhost` in Windows
- Need to use Windows host IP from WSL

## Recommended Approach

**Use Windows Python** (Option 1) - simplest and most reliable!

```cmd
trd_framework\GENERATE_1K_WINDOWS.bat
```

This avoids all networking issues.

## Verify Ollama is Running

```cmd
# In Windows PowerShell
curl http://localhost:11434/api/tags

# Should return JSON with model list
```

## Summary

| Approach | Pros | Cons |
|----------|------|------|
| **Windows Python** | ✅ No networking issues<br>✅ Simple<br>✅ Reliable | Requires Windows Python |
| **WSL with host IP** | ✅ Uses WSL | ❌ Need to find host IP<br>❌ IP may change |
| **Run Ollama in WSL** | ✅ All in WSL | ❌ Complex setup<br>❌ GPU passthrough issues |

**Recommendation**: Use `GENERATE_1K_WINDOWS.bat` for simplicity!
