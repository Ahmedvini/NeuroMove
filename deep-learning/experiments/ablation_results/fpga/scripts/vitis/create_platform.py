"""
create_platform.py — Vitis Unified IDE 2025.2 Python script that builds the
embedded software platform from the Vivado .xsa we exported.

Run with:
    vitis -s scripts/vitis/create_platform.py

Resilient to the known Vitis 2025.2 quirk where create_platform_component()
sometimes drops its gRPC connection right after the server finishes the SDT
generation phase. We catch that, then re-attach to the platform via
get_component() and continue.
"""

import os
import sys
import time

import vitis


REPO_ROOT = "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main"
WORKSPACE = os.path.join(REPO_ROOT, "build", "vitis_workspace")
XSA_PATH = os.path.join(REPO_ROOT, "build", "db_atcnet_zcu104.xsa")
PLATFORM_NAME = "db_atcnet_zcu104_platform"
DOMAIN_NAME = "standalone_psu_cortexa53_0"
CPU = "psu_cortexa53_0"


def get_or_create_platform(client):
    """Create the platform, or grab it if a prior gRPC drop already made it."""
    try:
        print(f"[create_platform] creating platform '{PLATFORM_NAME}'...")
        platform = client.create_platform_component(
            name=PLATFORM_NAME,
            hw_design=XSA_PATH,
        )
        print("[create_platform] create_platform_component returned cleanly")
        return platform
    except Exception as exc:
        print(f"[create_platform] create raised: {exc}")
        # Server may have completed the creation but the gRPC channel died.
        # Wait a moment for the server to settle, then re-attach.
        time.sleep(3)
        print(f"[create_platform] attempting get_component('{PLATFORM_NAME}')...")
        platform = client.get_component(name=PLATFORM_NAME)
        if platform is None:
            raise RuntimeError("get_component returned None; platform truly not created.")
        print("[create_platform] re-attached to platform handle")
        return platform


def main() -> int:
    if not os.path.isfile(XSA_PATH):
        print(f"ERROR: {XSA_PATH} not found. Run write_hw_platform in Vivado first.")
        return 1

    os.makedirs(WORKSPACE, exist_ok=True)

    print(f"[create_platform] workspace: {WORKSPACE}")
    print(f"[create_platform] .xsa:      {XSA_PATH}")

    client = vitis.create_client()
    client.set_workspace(path=WORKSPACE)

    platform = get_or_create_platform(client)

    # Report to confirm we have a live handle
    try:
        platform.report()
    except Exception as exc:
        print(f"[create_platform] platform.report() failed: {exc}")

    print(f"[create_platform] adding standalone domain '{DOMAIN_NAME}' on {CPU}...")
    try:
        platform.add_domain(name=DOMAIN_NAME, cpu=CPU, os="standalone")
    except Exception as exc:
        print(f"[create_platform] add_domain raised: {exc} -- checking if domain already exists")
    try:
        platform.list_domains()
    except Exception as exc:
        print(f"[create_platform] list_domains failed: {exc}")

    print("[create_platform] building platform (3-5 min)...")
    try:
        status = platform.build()
        print(f"[create_platform] build status: {status}")
    except Exception as exc:
        print(f"[create_platform] platform.build raised: {exc}")
        return 1

    xpfm = os.path.join(
        WORKSPACE, PLATFORM_NAME, "export", PLATFORM_NAME, f"{PLATFORM_NAME}.xpfm"
    )
    if os.path.isfile(xpfm):
        print(f"[create_platform] SUCCESS. xpfm at: {xpfm}")
        return 0
    print(f"[create_platform] WARN: xpfm not found at {xpfm}; check workspace.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
