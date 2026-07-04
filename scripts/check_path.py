import os
import sys
import shutil

print("PATH:")
for p in os.environ.get("PATH", "").split(os.pathsep):
    print("  " + p)

print("\nLooking for powershell:")
print(shutil.which("powershell"))

print("\nLooking for npx:")
print(shutil.which("npx"))
print(shutil.which("node"))
