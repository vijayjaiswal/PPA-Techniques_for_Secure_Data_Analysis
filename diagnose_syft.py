import traceback
import sys
import os

print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import syft as sy
    print("Syft imported successfully!")
    print(f"Syft version: {sy.__version__}")
except Exception:
    print("-" * 40)
    print("CRITICAL ERROR DURING IMPORT:")
    traceback.print_exc()
    print("-" * 40)
