param(
  [string]$CJson = "work\src\train\unified_jsons",
  [string]$FJson = "F:\work\src\train\unified_jsons",
  [int]$IntervalSec = 20,
  [int]$SafeAgeSec  = 60
)

$ErrorActionPreference = "Stop"
function Ensure-Dir([string]$p){ if(-not (Test-Path $p)){ New-Item -ItemType Directory -Force | Out-Null } }
function Test-FileStable([string]$p,[int]$ms=1500){
  if(-not (Test-Path $p)) { return $false }
  try{ $a=(Get-Item $p).Length; Start-Sleep -Milliseconds $ms; if(-not (Test-Path $p)){return $false}; $b=(Get-Item $p).Length; return ($a -eq $b) } catch { return $false }
}
function Move-Safe([string]$src,[string]$dstDir){
  if(-not (Test-Path $src)) { return $false }
  if(-not (Test-FileStable $src)) { return $false }
  Ensure-Dir $dstDir
  try { Move-Item -LiteralPath $src -Destination $dstDir -Force; return $true }
  catch { $base = Split-Path $src -Leaf; $srcDir = Split-Path $src -Parent; $null = robocopy $srcDir $dstDir $base /MOV /R:1 /W:0; return ($LastExitCode -lt 8) }
}

Ensure-Dir $FJson
Write-Host "[move-unified] C:$CJson → F:$FJson (every $IntervalSec s)"
while ($true) {
  if(Test-Path $CJson){
    $cutoff = (Get-Date).AddSeconds(-$SafeAgeSec)
    Get-ChildItem $CJson -File -Filter *.json -EA SilentlyContinue |
      Where-Object { $_.LastWriteTime -lt $cutoff } |
      ForEach-Object { [void](Move-Safe $_.FullName $FJson) }
  }
  $cFree = (Get-PSDrive C).Free/1GB
  Write-Host ("[move-unified] F count={0}  C:\ free={1:n1} GB  {2}" -f ((Get-ChildItem $FJson -Filter *.json -EA SilentlyContinue).Count),$cFree,(Get-Date).ToString("HH:mm:ss"))
  Start-Sleep -Seconds $IntervalSec
}
