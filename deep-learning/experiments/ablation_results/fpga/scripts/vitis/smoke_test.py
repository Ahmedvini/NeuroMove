"""smoke_test.py — try create_platform_component with AMD's own example .xsa.

If this succeeds, the bug is specific to our db_atcnet_zcu104.xsa.
If this fails the same way, Vitis 2025.2 install is broken on this machine.
"""
import os
import vitis

WORKSPACE = "/tmp/vitis_smoke_ws"
XSA = "/tools/2025.2/Vitis/scripts/python_pkg/hsi/examples/test_01/zcu106.xsa"

os.makedirs(WORKSPACE, exist_ok=True)
client = vitis.create_client()
client.set_workspace(path=WORKSPACE)

print(f"[smoke] creating platform from {XSA}")
try:
    plat = client.create_platform_component(name="smoke_kc705", hw_design=XSA)
    print(f"[smoke] SUCCESS: {plat}")
except Exception as e:
    print(f"[smoke] FAILED: {e}")
