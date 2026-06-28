"""
create_app.py — creates an application component on the platform produced by
create_platform.py, drops sw/zynq_ps/main.c into its src/, and builds it.

Run with:
    vitis -s scripts/vitis/create_app.py

Prerequisite: create_platform.py has been run successfully and the platform
.xpfm exists.
"""

import os
import shutil
import sys

import vitis


REPO_ROOT = "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main"
WORKSPACE = os.path.join(REPO_ROOT, "build", "vitis_workspace")
PLATFORM_NAME = "db_atcnet_zcu104_platform"
DOMAIN_NAME = "standalone_psu_cortexa53_0"
APP_NAME = "db_atcnet_ps_app"
MAIN_C_SRC = os.path.join(REPO_ROOT, "sw", "zynq_ps", "main.c")

PLATFORM_XPFM = os.path.join(
    WORKSPACE,
    PLATFORM_NAME,
    "export",
    PLATFORM_NAME,
    f"{PLATFORM_NAME}.xpfm",
)


def main() -> int:
    if not os.path.isfile(PLATFORM_XPFM):
        print(f"ERROR: platform xpfm not found at {PLATFORM_XPFM}")
        print("Run create_platform.py first.")
        return 1
    if not os.path.isfile(MAIN_C_SRC):
        print(f"ERROR: {MAIN_C_SRC} not found.")
        return 1

    client = vitis.create_client()
    client.set_workspace(path=WORKSPACE)

    print(f"[create_app] creating app '{APP_NAME}' on platform '{PLATFORM_NAME}'...")
    app = client.create_app_component(
        name=APP_NAME,
        platform=PLATFORM_XPFM,
        domain=DOMAIN_NAME,
        template="empty_application",
    )

    app_src_dir = os.path.join(WORKSPACE, APP_NAME, "src")
    os.makedirs(app_src_dir, exist_ok=True)
    dest = os.path.join(app_src_dir, "main.c")
    print(f"[create_app] copying {MAIN_C_SRC} -> {dest}")
    shutil.copy(MAIN_C_SRC, dest)

    # Remove the auto-generated stub helloworld.c if it exists so we don't
    # have two main()s.
    stub = os.path.join(app_src_dir, "helloworld.c")
    if os.path.isfile(stub):
        print(f"[create_app] removing stub {stub}")
        os.remove(stub)

    print("[create_app] building app...")
    status = app.build()
    print(f"[create_app] build status: {status}")

    elf = os.path.join(WORKSPACE, APP_NAME, "build", f"{APP_NAME}.elf")
    if os.path.isfile(elf):
        print(f"[create_app] SUCCESS. ELF at: {elf}")
    else:
        print(f"[create_app] WARN: expected ELF at {elf} not found; check build log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
