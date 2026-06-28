# build_app.tcl — xsct script: build the embedded platform + app from .xsa
#
# Run with:
#   source /tools/2025.2/Vitis/settings64.sh
#   xsct /home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/scripts/vitis/build_app.tcl
#
# Does the full bring-up:
#   1. set workspace
#   2. create platform from .xsa (bare-metal, A53 core 0)
#   3. generate BSP + FSBL
#   4. create empty app on the platform
#   5. import sw/zynq_ps/main.c into the app
#   6. build the app -> .elf

set REPO_ROOT      "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main"
set WORKSPACE      "$REPO_ROOT/build/vitis_workspace"
set XSA            "$REPO_ROOT/build/db_atcnet_zcu104.xsa"
set MAIN_C         "$REPO_ROOT/sw/zynq_ps/main.c"
set PLATFORM_NAME  "db_atcnet_zcu104_platform"
set APP_NAME       "db_atcnet_ps_app"
set DOMAIN_NAME    "standalone_psu_cortexa53_0"

puts "==[clock format [clock seconds]] starting xsct build flow=="

# 1. Workspace
file mkdir $WORKSPACE
setws $WORKSPACE
puts "[xsct] workspace = [getws]"

# 2. Platform
puts "[xsct] creating platform '$PLATFORM_NAME' from $XSA"
platform create -name $PLATFORM_NAME \
                -hw $XSA \
                -proc psu_cortexa53_0 \
                -os standalone \
                -fsbl-target psu_cortexa53_0 \
                -out $WORKSPACE

# Make sure we're working on this platform
platform active $PLATFORM_NAME

# 3. Build / generate the platform (compiles FSBL + BSP)
puts "[xsct] generating platform (this is the slow step, ~3-5 min)..."
platform generate

# 4. Application
puts "[xsct] creating app '$APP_NAME' on $PLATFORM_NAME/$DOMAIN_NAME"
app create -name $APP_NAME \
           -platform $PLATFORM_NAME \
           -domain $DOMAIN_NAME \
           -template "Empty Application(C)"

# 5. Import main.c
puts "[xsct] importing $MAIN_C"
importsources -name $APP_NAME -path $MAIN_C

# Remove the auto-generated stub if it exists so we don't have two main()s
set stub_path "$WORKSPACE/$APP_NAME/src/helloworld.c"
if {[file exists $stub_path]} {
    puts "[xsct] removing stub $stub_path"
    file delete $stub_path
}

# 6. Build
puts "[xsct] building '$APP_NAME'..."
app build -name $APP_NAME

# Confirm
set elf "$WORKSPACE/$APP_NAME/Debug/$APP_NAME.elf"
if {[file exists $elf]} {
    puts "[xsct] SUCCESS -- elf at $elf"
    puts "[xsct] size: [file size $elf] bytes"
} else {
    puts "[xsct] WARN: expected ELF at $elf not found; check $WORKSPACE/$APP_NAME/Debug/ for actual location"
}

puts "==[clock format [clock seconds]] done=="
exit
