# Builds hashfinder, fastfinder (gcc + zlib) and gpufinder (nvcc + MSVC). Usage: powershell -File build.ps1 [-NoGpu]
param([switch]$NoGpu)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== hashfinder / fastfinder (gcc)"
gcc -O3 -march=native -Wall -Wextra -o fastfinder.exe fastfinder.c -lz -lpthread
if ($LASTEXITCODE) { throw "fastfinder build failed" }
gcc -O3 -march=native -Wall -Wextra -o hashfinder.exe hashfinder.c -lz -lpthread
if ($LASTEXITCODE) { throw "hashfinder build failed" }

if ($NoGpu) { return }
Write-Host "== gpufinder (nvcc, sm_120)"
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) { $nvcc = $nvcc.Source } else {
    $nvcc = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $nvcc) { Write-Warning "nvcc not found, skipping gpufinder"; return }
$vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $vcvars) { Write-Warning "vcvars64.bat (Visual Studio 2022) not found, skipping gpufinder"; return }
cmd /c "`"$vcvars`" >nul && `"$nvcc`" -O3 -arch=sm_120 -o gpufinder.exe gpufinder.cu"
if ($LASTEXITCODE) { throw "gpufinder build failed" }
Remove-Item gpufinder.lib, gpufinder.exp -ErrorAction SilentlyContinue
Write-Host "built: fastfinder.exe hashfinder.exe gpufinder.exe"
