param(
  [ValidateSet('train','valid','test')]
  [string]$Split = 'train',
  [string]$RootF = 'F:\work\src',
  [int]$IntervalSec = 60,
  [switch]$SkipAug           # set -SkipAug to skip augmentation and only do hetero+labels
)

$ErrorActionPreference = 'Stop'

# Paths on F:
$FJson  = Join-Path $RootF "$Split\unified_jsons"
$FAug   = Join-Path $RootF "$Split\unified_aug"
$FReady = Join-Path $RootF "$Split\hetero_ready"
$FLogs  = Join-Path $RootF "$Split\logs"
New-Item -ItemType Directory -Force $FAug,$FReady,$FLogs | Out-Null

function Get-Bases($dir,$pattern){
  if(Test-Path $dir){
    Get-ChildItem $dir -File -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object { $_.BaseName }
  } else { @() }
}
function Count-Files($dir,$pattern){
  if(Test-Path $dir){ (Get-ChildItem $dir -File -Filter $pattern -ErrorAction SilentlyContinue).Count } else { 0 }
}

Write-Host "[downstream:$Split] FJson=$FJson  FAug=$FAug  FReady=$FReady"
Write-Host "[downstream:$Split] Loop every $IntervalSec s. Press Ctrl+C to stop."

while ($true) {
  # --- snapshot of what exists now ---
  $jsonBases = Get-Bases $FJson  '*.json'   # ground truth set
  $augBases  = Get-Bases $FAug   '*.json'
  $ptBases   = Get-Bases $FReady '*.pt'

  # backlog
  $toAug = $jsonBases | Where-Object { $augBases -notcontains $_ }
  $toHet = $augBases  | Where-Object { $ptBases -notcontains $_ }

  # progress numbers
  $total   = $jsonBases.Count
  $augDone = $augBases.Count
  $hetDone = $ptBases.Count

  $pctAug = [int](100 * $augDone / [math]::Max(1,$total))
  $pctHet = [int](100 * $hetDone / [math]::Max(1,$total))

  # --- progress bars ---
  Write-Progress -Id 1 -Activity "Augment → $Split (on F:)"      -Status "$augDone / $total augmented" -PercentComplete $pctAug
  Write-Progress -Id 2 -Activity "Hetero+Labels → $Split (on F:)" -Status "$hetDone / $total .pt"       -PercentComplete $pctHet

  # --- do work (idempotent; safe to repeat) ---
  if(-not $SkipAug -and $toAug.Count -gt 0 -and (Test-Path $FJson)){
    python .\tools\augment_and_load_pyg.py --inp $FJson --out $FAug --limit 99999999 `
      1>> (Join-Path $FLogs "augment.out.log") 2>> (Join-Path $FLogs "augment.err.log")
  }

  if($toHet.Count -gt 0 -and (Test-Path $FAug)){
    python .\tools\tag_and_reverse_pyg.py --inp $FAug --out $FReady --params-as-sources `
      1>> (Join-Path $FLogs "tag.out.log") 2>> (Join-Path $FLogs "tag.err.log")

    python .\tools\label_dir.py --inp $FReady `
      1>> (Join-Path $FLogs "label.out.log") 2>> (Join-Path $FLogs "label.err.log")
  }

  # heartbeat line (helpful if the console scrolls)
  $nJ = Count-Files $FJson  *.json
  $nA = Count-Files $FAug   *.json
  $nP = Count-Files $FReady *.pt
  Write-Host ("[{0}] json={1}  aug={2}  .pt={3}  backlog: toAug={4} toHet={5}  {6}" -f $Split,$nJ,$nA,$nP,$toAug.Count,$toHet.Count,(Get-Date).ToString("HH:mm:ss"))

  Start-Sleep -Seconds $IntervalSec
}
