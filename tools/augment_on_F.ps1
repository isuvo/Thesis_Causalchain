# tools\augment_on_F.ps1  (TRAIN)
param([string]$FJson="F:\work\src\train\unified_jsons",
      [string]$FAug="F:\work\src\train\unified_aug",
      [string]$FLogs="F:\work\src\train\logs",
      [int]$IntervalSec=60)

$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force $FAug,$FLogs | Out-Null

while($true){
  $j = (Get-ChildItem $FJson -File -Filter *.json -EA SilentlyContinue).Count
  $a = (Get-ChildItem $FAug  -File -Filter *.json -EA SilentlyContinue).Count
  $pct = [int](100 * $a / [math]::Max(1,$j))
  Write-Progress -Activity "Augment → train (on F:)" -Status "$a / $j augmented" -PercentComplete $pct

  if(Test-Path $FJson){
    python .\tools\augment_and_load_pyg.py --inp $FJson --out $FAug --limit 99999999 `
      1>> (Join-Path $FLogs "augment.out.log") 2>> (Join-Path $FLogs "augment.err.log")
  }
  Start-Sleep -Seconds $IntervalSec
}
