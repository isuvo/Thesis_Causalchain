param(
  [string]$FAug   = "F:\work\src\train\unified_aug",
  [string]$FReady = "F:\work\src\train\hetero_ready",
  [string]$FLogs  = "F:\work\src\train\logs",
  [string]$Tmp    = "F:\work\src\train\_tmp_hetero",
  [string]$Quarantine = "F:\work\src\train\bad_aug",
  [int]$TimeoutSec = 600   # per-file safety timeout
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $FReady,$FLogs,$Tmp,$Quarantine | Out-Null

# pending = augmented JSONs without a .pt yet
$augBases = if (Test-Path $FAug) { Get-ChildItem $FAug -File -Filter *.json -EA SilentlyContinue | % { $_.BaseName } } else { @() }
$ptBases  = if (Test-Path $FReady) { Get-ChildItem $FReady -File -Filter *.pt -EA SilentlyContinue | % { $_.BaseName } } else { @() }
$pending  = $augBases | Where-Object { $ptBases -notcontains $_ }

$total = $pending.Count; $i = 0
foreach ($base in $pending) {
  $i++; $pct = [int](100 * $i / [math]::Max(1,$total))
  Write-Progress -Activity "Hetero build (per-file)" -Status "$i / $total  ($base)" -PercentComplete $pct

  $json = Join-Path $FAug   ($base + ".json")
  $pt   = Join-Path $FReady ($base + ".pt")
  if (-not (Test-Path $json)) { continue }
  if (Test-Path $pt) { continue } # already done

  # isolate this shard in a private temp dir so failures can't block others
  $tmpDir = Join-Path $Tmp $base
  if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
  New-Item -ItemType Directory -Force $tmpDir | Out-Null
  Copy-Item $json -Destination $tmpDir -Force

  $outLog = Join-Path $FLogs ("tag_" + $base + ".out.log")
  $errLog = Join-Path $FLogs ("tag_" + $base + ".err.log")

  # run tag+reverse just for this one file
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "python"
  $psi.Arguments = ".\tools\tag_and_reverse_pyg.py --inp `"$tmpDir`" --out `"$FReady`" --params-as-sources"
  $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true
  $p = New-Object System.Diagnostics.Process
  $p.StartInfo = $psi
  $null = $p.Start()
  $ok = $p.WaitForExit($TimeoutSec * 1000)
  if (-not $ok) { try { $p.Kill() } catch {} }

  $p.StandardOutput.ReadToEnd() | Out-File -Encoding utf8 -FilePath $outLog
  $p.StandardError.ReadToEnd() | Out-File -Encoding utf8 -FilePath $errLog

  if ($ok -and $p.ExitCode -eq 0 -and (Test-Path $pt)) {
    # success
    Remove-Item $tmpDir -Recurse -Force
  } else {
    # quarantine the problematic JSON so the pipeline keeps moving
    Move-Item $json (Join-Path $Quarantine ($base + ".json")) -Force
    Write-Warning "FAILED: $base  (moved to $Quarantine; see $errLog)"
  }
}

Write-Host "Done. Built $( (Get-ChildItem $FReady -File -Filter *.pt -EA SilentlyContinue).Count ) .pt in $FReady"
