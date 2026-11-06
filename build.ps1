# Builds hashfinder, fastfinder, namemcfinder (gcc + zlib) and gpufinder, namemcgpu, namemccollide (nvcc + MSVC). Usage: powershell -File build.ps1 [-NoGpu]
param([switch]$NoGpu)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== hashfinder / fastfinder / namemcfinder (gcc)"
gcc -O3 -march=native -Wall -Wextra -o fastfinder.exe fastfinder.c -lz -lpthread
if ($LASTEXITCODE) { throw "fastfinder build failed" }
gcc -O3 -march=native -Wall -Wextra -o hashfinder.exe hashfinder.c -lz -lpthread
if ($LASTEXITCODE) { throw "hashfinder build failed" }
gcc -O3 -march=native -Wall -Wextra -o namemcfinder.exe namemcfinder.c -lz -lpthread
if ($LASTEXITCODE) { throw "namemcfinder build failed" }

if ($NoGpu) { return }
Write-Host "== gpufinder / namemcgpu (nvcc, sm_120)"
# Newest installed toolkit first: CUDA 12.8's ptxas ran for 40+ minutes on the stat kernels without finishing; 13.0 builds them.
$nvcc = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe" -ErrorAction SilentlyContinue |
    Sort-Object { [version]($_.FullName -replace '.*\\CUDA\\v([\d.]+)\\.*', '$1') } -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $nvcc) { $c = Get-Command nvcc -ErrorAction SilentlyContinue; if ($c) { $nvcc = $c.Source } }
if ($nvcc) { Write-Host "using $nvcc" }
if (-not $nvcc) { Write-Warning "nvcc not found, skipping gpufinder and namemcgpu"; return }
$vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $vcvars) { Write-Warning "vcvars64.bat (Visual Studio 2022) not found, skipping gpufinder and namemcgpu"; return }
cmd /c "`"$vcvars`" >nul && `"$nvcc`" -O3 -arch=sm_120 -o gpufinder.exe gpufinder.cu"
if ($LASTEXITCODE) { throw "gpufinder build failed" }
cmd /c "`"$vcvars`" >nul && `"$nvcc`" -O3 -arch=sm_120 -o namemcgpu.exe namemcgpu.cu"
if ($LASTEXITCODE) { throw "namemcgpu build failed" }
cmd /c "`"$vcvars`" >nul && `"$nvcc`" -O3 -arch=sm_120 -o namemccollide.exe namemccollide.cu"
if ($LASTEXITCODE) { throw "namemccollide build failed" }
Remove-Item gpufinder.lib, gpufinder.exp, namemcgpu.lib, namemcgpu.exp, namemccollide.lib, namemccollide.exp -ErrorAction SilentlyContinue
Write-Host "built: fastfinder.exe hashfinder.exe namemcfinder.exe gpufinder.exe namemcgpu.exe namemccollide.exe"
