param(
  [string]$SourceRoot = "work\src",      # on C:
  [string]$DestRoot   = "F:\work\src",   # on F:
  [int]$IntervalSec   = 30,              # how often to scan
  [int]$SafeAgeSec    = 90               # minimum age before moving
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p){
  if(-not (Test-Path $p)){ New-Item -ItemType Directory -Force $p | Out-Null }
}

function Test-FileStable([string]$p,[int]$ms=1500){
  if(-not (Test-Path $p)) { return $false }
  try{
    $a = (Get-Item $p).Length
    Start-Sleep -Milliseconds $ms
    if(-not (Test-Path $p)) { return $false }
    $b = (Get-Item $p).Length
    return ($a -eq $b)
  } catch { return $false }
}

function Move-Safe([string]$src,[string]$dstDir){
  if(-not (Test-Path $src)) { return $false }
  if(-not (Test-FileStable $src)) { return $false }
  Ensure-Dir $dstDir
  try {
    Move-Item -LiteralPath $src -Destination $dstDir -Force
    return $true
  } catch { return $false }
}

Ensure-Dir $DestRoot
$splits = @("train","valid","test")
Write-Host "[spill] Watching: $SourceRoot  →  $DestRoot  (every $IntervalSec s; safe age $SafeAgeSec s)"

while ($true) {
  $cutoff = (Get-Date).AddSeconds(-$SafeAgeSec)  # <-- compute once per scan
  $movedJSON=0; $movedAUG=0; $movedPT=0

  foreach($sp in $splits){
    $cSplit = Join-Path $SourceRoot $sp
    if(-not (Test-Path $cSplit)) { continue }

    $cJson   = Join-Path $cSplit "unified_jsons"
    $cAug    = Join-Path $cSplit "unified_aug"
    $cReady  = Join-Path $cSplit "hetero_ready"

    $fSplit  = Join-Path $DestRoot $sp
    $fJson   = Join-Path $fSplit "unified_jsons"
    $fAug    = Join-Path $fSplit "unified_aug"
    $fReady  = Join-Path $fSplit "hetero_ready"

    Ensure-Dir $fJson; Ensure-Dir $fAug; Ensure-Dir $fReady

    # 1) Move .pt as soon as it's stable (final products)
    if(Test-Path $cReady){
      Get-ChildItem $cReady -File -Filter *.pt -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
          if(Move-Safe $_.FullName $fReady){ $movedPT++ }
        }
    }


    # 1b) Also move any .pt saved in unified_aug (diagnostic snapshots)
    if(Test-Path $cAug){
    Get-ChildItem $cAug -File -Filter *.pt -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
        if(Move-Safe $_.FullName $fAug){ $movedAUG++ }
        }
    }

    # 2) Move unified_aug JSON if corresponding .pt exists (on C or already moved to F)
    if(Test-Path $cAug){
      Get-ChildItem $cAug -File -Filter *.json -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
          $base = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
          $ptC  = Join-Path $cReady ($base + ".pt")
          $ptF  = Join-Path $fReady ($base + ".pt")
          if( (Test-Path $ptC) -or (Test-Path $ptF) ){
            if(Move-Safe $_.FullName $fAug){ $movedAUG++ }
          }
        }
    }

    # 3) Move unified_json only after unified_aug exists (on C or F)
    if(Test-Path $cJson){
      Get-ChildItem $cJson -File -Filter *.json -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
          $base = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
          $augC = Join-Path $cAug  ($base + ".json")
          $augF = Join-Path $fAug  ($base + ".json")
          if( (Test-Path $augC) -or (Test-Path $augF) ){
            if(Move-Safe $_.FullName $fJson){ $movedJSON++ }
          }
        }
    }
  }

  $cFree = (Get-PSDrive C).Free/1GB
  Write-Host ("[spill] moved: PT={0}  AUG={1}  JSON={2}   C:\ free={3:n1} GB  {4}" -f $movedPT,$movedAUG,$movedJSON,$cFree,(Get-Date).ToString("HH:mm:ss"))
  Start-Sleep -Seconds $IntervalSec
}
