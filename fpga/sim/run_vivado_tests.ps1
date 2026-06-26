
param(
    [Parameter(Position=0)]
    [ValidateSet("temporal", "top", "all", "hmac", "aes", "help")]
    [string]$TestName = "help"
)

#Define the project root directory

$ProjectRoot = "C:\College\Graduation\FPGA\fpga-atcnet"

function Show-Help {
    Write-Host ""
    Write-Host "ATCNet Vivado Test Runner" -ForegroundColor Cyan
    Write-Host "=========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\run_vivado_tests.ps1 [test_name]"
    Write-Host ""
    Write-Host "Available Tests:" -ForegroundColor Yellow
    Write-Host "  temporal  - Run temporal_conv unit test"
    Write-Host "  top       - Run top-level integration test"
    Write-Host "  all       - Run comprehensive test suite (all modules)"
    Write-Host "  hmac      - Run HMAC-SHA-256 + SHA-256 + RSA Secure Boot security test"
    Write-Host "  aes       - Run AES-256-GCM security test"
    Write-Host "  help      - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_vivado_tests.ps1 temporal"
    Write-Host "  .\run_vivado_tests.ps1 top"
    Write-Host "  .\run_vivado_tests.ps1 all"
    Write-Host "  .\run_vivado_tests.ps1 hmac"
    Write-Host "  .\run_vivado_tests.ps1 aes"
    Write-Host ""
}

function Run-TemporalTest {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Running Temporal Conv Unit Test" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    
    Push-Location $ProjectRoot
    
    Write-Host "[1/3] Compiling RTL and testbench..." -ForegroundColor Yellow
    xvlog --sv rtl/conv/temporal_conv.sv sim/temporal_conv_tb.sv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed!" -ForegroundColor Red
        Pop-Location
        return
    }
    
    Write-Host "[2/3] Elaborating design..." -ForegroundColor Yellow
    xelab temporal_conv_tb -debug typical
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Elaboration failed!" -ForegroundColor Red
        Pop-Location
        return
    }
    
    Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
    xsim temporal_conv_tb -runall
    
    Pop-Location
    Write-Host ""
    Write-Host "Test completed!" -ForegroundColor Green
}

function Run-TopTest {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Running Top-Level Integration Test" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    
    Push-Location $ProjectRoot
    
    Write-Host "[1/3] Compiling RTL and testbench..." -ForegroundColor Yellow
    xvlog --sv rtl/conv/temporal_conv.sv `
                 rtl/conv/dual_branch_conv.sv `
                 rtl/conv/channel_scale.sv `
                 rtl/window/window_gen.sv `
                 rtl/window/window_reader.sv `
                 rtl/top.sv `
                 sim/top_tb.sv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed!" -ForegroundColor Red
        Pop-Location
        return
    }
    
    Write-Host "[2/3] Elaborating design..." -ForegroundColor Yellow
    xelab top_tb -debug typical
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Elaboration failed!" -ForegroundColor Red
        Pop-Location
        return
    }
    
    Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
    xsim top_tb -runall
    
    Pop-Location
    Write-Host ""
    Write-Host "Test completed!" -ForegroundColor Green
}

function Run-AllTests {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Running Comprehensive Test Suite (All Modules)" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    
    Push-Location $ProjectRoot
    
    Write-Host "[1/3] Compiling all RTL and testbench..." -ForegroundColor Yellow
    xvlog --sv rtl/conv/temporal_conv.sv `
                 rtl/conv/dual_branch_conv.sv `
                 rtl/conv/channel_scale.sv `
                 rtl/window/window_gen.sv `
                 rtl/window/window_reader.sv `
                 rtl/conv/spatial_conv_serial_stream.sv `
                 rtl/fusion/temporal_fusion.sv `
                 rtl/attention/temporal_multihead_attention.sv `
                 sim/all_modules_tb.sv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed!" -ForegroundColor Red
        Pop-Location
        return
    }
    
    Write-Host "[2/3] Elaborating design..." -ForegroundColor Yellow
    xelab all_modules_tb -debug typical
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Elaboration failed!" -ForegroundColor Red
        Pop-Location
        return
    }
    
    Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
    xsim all_modules_tb -runall
    
    Pop-Location
    Write-Host ""
    Write-Host "Test completed!" -ForegroundColor Green
}

function Run-HmacTest {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Running HMAC-SHA-256 Security Test" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""

    Push-Location $ProjectRoot

    Write-Host "[1/3] Compiling RTL and testbench..." -ForegroundColor Yellow
    xvlog --sv rtl/security/sha256_core.sv `
                 rtl/security/sha256_hash_chain.sv `
                 rtl/security/hmac_sha256.sv `
                 rtl/security/gcm_ghash.sv `
                 rtl/security/aes_256_core.sv `
                 rtl/security/aes_256_gcm.sv `
                 rtl/security/rsa2048_core.sv `
                 rtl/security/secure_boot.sv `
                 rtl/security/eeg_security_top.sv `
                 sim/hmac_sha256_tb.sv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed!" -ForegroundColor Red
        Pop-Location
        return
    }

    Write-Host "[2/3] Elaborating design..." -ForegroundColor Yellow
    xelab hmac_sha256_tb -debug typical
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Elaboration failed!" -ForegroundColor Red
        Pop-Location
        return
    }

    Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
    xsim hmac_sha256_tb -runall

    Pop-Location
    Write-Host ""
    Write-Host "HMAC-SHA-256 test completed!" -ForegroundColor Green
}

function Run-AesTest {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Running AES-256-GCM Security Test" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""

    Push-Location $ProjectRoot

    Write-Host "[1/3] Compiling RTL and testbench..." -ForegroundColor Yellow
    xvlog --sv rtl/security/gcm_ghash.sv `
                 rtl/security/aes_256_core.sv `
                 rtl/security/aes_256_gcm.sv `
                 sim/aes_256_gcm_tb.sv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed!" -ForegroundColor Red
        Pop-Location
        return
    }

    Write-Host "[2/3] Elaborating design..." -ForegroundColor Yellow
    xelab aes_256_gcm_tb -debug typical
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Elaboration failed!" -ForegroundColor Red
        Pop-Location
        return
    }

    Write-Host "[3/3] Running simulation..." -ForegroundColor Yellow
    xsim aes_256_gcm_tb -runall

    Pop-Location
    Write-Host ""
    Write-Host "AES-256-GCM test completed!" -ForegroundColor Green
}

# Main execution
switch ($TestName) {
    "temporal" { Run-TemporalTest }
    "top"      { Run-TopTest }
    "all"      { Run-AllTests }
    "hmac"     { Run-HmacTest }
    "aes"      { Run-AesTest }
    "help"     { Show-Help }
    default    { Show-Help }
}
