param(
  [string]$FJson = "F:\work\src\train\unified_jsons",
  [string]$FAug  = "F:\work\src\train\unified_aug",
  [string]$FLogs = "F:\work\src\train\logs",
  [string]$Tmp   = "F:\work\src\train\_tmp_augment",
  [string]$Quarantine = "F:\work\src\train\bad_json",
  [int]$TimeoutSec = 600
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $FAug,$FLogs,$Tmp,$Quarantine | Out-Null

function Bases($dir,$pat){ if(Test-Path $dir){ Get-ChildItem $dir -File -Filter $pat -ea SilentlyContinue | %{$_.BaseName} } else {@()} }

# backlog = JSON without augmented JSON
$jsonBases = Bases $FJson  *.json
$augBases  = Bases $FAug   *.json
$pending   = $jsonBases | Where-Object { $augBases -notcontains $_ }

$total = $pending.Count; $i = 0
foreach($base in $pending){
  $i++; $pct = [int](100 * $i / [math]::Max(1,$total))
  Write-Progress -Activity "Augment (per-file)" -Status "$i / $total  ($base)" -PercentComplete $pct

  $src = Join-Path $FJson ($base + ".json")
  $dst = Join-Path $FAug  ($base + ".json")
  if(-not (Test-Path $src)){ continue }
  if(Test-Path $dst){ continue }

  # isolate to a temp dir so the python tool processes only this file
  $td = Join-Path $Tmp $base
  if(Test-Path $td){ Remove-Item $td -Recurse -Force }
  New-Item -ItemType Directory -Force $td | Out-Null
  Copy-Item $src -Destination $td -Force

  $outLog = Join-Path $FLogs ("augment_" + $base + ".out.log")
  $errLog = Join-Path $FLogs ("augment_" + $base + ".err.log")

  # launch python for just this file (dir with 1 file)
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "python"
  $psi.Arguments = ".\tools\augment_and_load_pyg.py --inp `"$td`" --out `"$FAug`" --limit 1"
  $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true

  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  $null = $p.Start()
  $ok = $p.WaitForExit($TimeoutSec * 1000)
  if(-not $ok){ try{$p.Kill()}catch{} }

  $p.StandardOutput.ReadToEnd() | Out-File -Encoding utf8 -FilePath $outLog
  $p.StandardError.ReadToEnd()  | Out-File -Encoding utf8 -FilePath $errLog

  if($ok -and $p.ExitCode -eq 0 -and (Test-Path $dst)){
    Remove-Item $td -Recurse -Force
  } else {
    # quarantine the original unified_json so the queue moves on
    Move-Item $src (Join-Path $Quarantine ($base + ".json")) -Force
    Write-Warning "AUGMENT FAILED: $base  (moved to $Quarantine; see $errLog)"
    Remove-Item $td -Recurse -Force
  }
}

# final heartbeat
$j = (Get-ChildItem $FJson -File -Filter *.json -ea SilentlyContinue).Count
$a = (Get-ChildItem $FAug  -File -Filter *.json -ea SilentlyContinue).Count
Write-Host "Done augmenting. F:\ json=$j  aug=$a"
