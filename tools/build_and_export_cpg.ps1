param(
  [ValidateSet('train','valid','test')][string]$Split = 'train',
  [int]$HeapGB = 8
)

$ErrorActionPreference = 'Stop'

# --- CONFIG ---
$JOERN      = 'C:\tools\joern'
$PROJECT    = (Get-Location).Path
$DATA_ROOT  = Join-Path $PROJECT 'data\dataset\ReposVul_c_cpp'
$JSONL_MAP  = @{
  'train' = 'train_c_cpp_repository2.jsonl'
  'valid' = 'valid_c_cpp_repository2.jsonl'
  'test'  = 'test_c_cpp_repository2.jsonl'
}
$JSONL = Join-Path $DATA_ROOT $JSONL_MAP[$Split]
$BINS_OUT   = Join-Path $PROJECT 'work\cpg\bins'
$JSON_OUT   = Join-Path $PROJECT 'work\cpg\json'
$EXPORT_SC  = Join-Path $PROJECT 'tools\export_cpg_env.sc'

# --- CHECKS ---
if (-not (Test-Path $JOERN)) { throw "Joern not found at $JOERN" }
if (-not (Test-Path (Join-Path $JOERN 'c2cpg.bat'))) { throw "c2cpg.bat not found under $JOERN" }
if (-not (Test-Path $JSONL)) { throw "JSONL not found: $JSONL" }
if (-not (Test-Path $EXPORT_SC)) { throw "Exporter script missing: $EXPORT_SC" }

New-Item -ItemType Directory -Force -Path $BINS_OUT, $JSON_OUT | Out-Null

Write-Host "== Using JSONL: $JSONL"
Write-Host "== Binaries -> $BINS_OUT"
Write-Host "== Exports  -> $JSON_OUT"
Write-Host "== Joern    -> $JOERN"
Write-Host ""

# --- helper: normalize a candidate path and test it ---
function Resolve-RepoPath([string]$p) {
  if (-not $p) { return $null }
  $p = $p -replace '/', '\'
  # Absolute path?
  if (Test-Path $p -PathType Container) { return (Resolve-Path $p).Path }
  # Relative to dataset root?
  $rel = Join-Path $DATA_ROOT $p
  if (Test-Path $rel -PathType Container) { return (Resolve-Path $rel).Path }
  return $null
}

# --- helper: extract path from a JSON object ---
function Get-RepoLocalPath($obj) {
  $keys = @(
    'local_path','localDir','repo_dir','repo_path','repo_local_path','repo_local_dir',
    'path','src','source_dir','source_root','abs_local_path','absolute_local_path'
  )
  foreach ($k in $keys) {
    if ($obj.PSObject.Properties.Name -contains $k) {
      $val = [string]$obj.$k
      $rp = Resolve-RepoPath $val
      if ($rp) { return $rp }
    }
  }
  # Sometimes JSONL stores under 'repo' or 'project' nested objects
  foreach ($maybe in @('repo','project')) {
    if ($obj.$maybe) {
      foreach ($k in $keys) {
        if ($obj.$maybe.PSObject.Properties.Name -contains $k) {
          $val = [string]$obj.$maybe.$k
          $rp = Resolve-RepoPath $val
          if ($rp) { return $rp }
        }
      }
    }
  }
  return $null
}

# --- parse JSONL ---
$repoSet = New-Object System.Collections.Generic.HashSet[string]
$lineNo = 0

# Read in chunks for big files
Get-Content -LiteralPath $JSONL -ReadCount 500 | ForEach-Object {
  foreach ($line in $_) {
    $lineNo++
    if ($line -match '\S') {
      try {
        $obj = $line | ConvertFrom-Json
        $p = Get-RepoLocalPath $obj
        if ($p) { [void]$repoSet.Add($p) }
      } catch {
        Write-Warning "Skipping malformed JSON at line $lineNo"
      }
    }
  }
}

# --- fallback: scan for plausible repos if none found ---
if ($repoSet.Count -eq 0) {
  Write-Warning "No repo paths found in JSONL; scanning under $DATA_ROOT for plausible repos..."
  # Heuristic: a folder that contains >= 10 C/C++ files somewhere below.
  $candidates = Get-ChildItem -Path $DATA_ROOT -Directory -Recurse |
    Where-Object {
      (Get-ChildItem -Path $_.FullName -Recurse -Include *.c,*.cc,*.cpp,*.h,*.hpp -ErrorAction SilentlyContinue | Measure-Object).Count -ge 10
    }
  foreach ($c in $candidates) { [void]$repoSet.Add((Resolve-Path $c.FullName).Path) }
}

if ($repoSet.Count -eq 0) { throw "Still no repositories found. Check your JSONL and where the local repos actually live." }

Write-Host ("== Discovered {0} repositories" -f $repoSet.Count)
$repos = $repoSet.ToArray() | Sort-Object

# --- Build one cpg.bin per repo ---
foreach ($repoPath in $repos) {
  $repoName = Split-Path $repoPath -Leaf
  $outDir   = Join-Path $BINS_OUT $repoName
  $outBin   = Join-Path $outDir 'cpg.bin'
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  Write-Host ">> c2cpg: $repoName"
  & (Join-Path $JOERN 'c2cpg.bat') "-J-Xmx${HeapGB}g" $repoPath --output $outBin
}

# --- Export per-method JSON for each cpg.bin ---
$bins = Get-ChildItem -Path $BINS_OUT -Recurse -Filter 'cpg.bin'
foreach ($bin in $bins) {
  $repoName = Split-Path $bin.DirectoryName -Leaf
  $env:CPG_PATH = $bin.FullName
  $env:OUT_DIR  = $JSON_OUT
  $env:LOG_PATH = Join-Path $JSON_OUT ("export_" + $repoName + ".log")

  Write-Host ">> export: $repoName"
  & (Join-Path $JOERN 'joern.bat') --script $EXPORT_SC

  Remove-Item -Recurse -Force 'workspace' -ErrorAction SilentlyContinue
}

Write-Host "`n== DONE =="
Write-Host "CPG binaries: $BINS_OUT"
Write-Host "Exports     : $JSON_OUT"
