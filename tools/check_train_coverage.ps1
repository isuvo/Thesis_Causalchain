param(
  [string]$TrainRoot = "work\src\train",
  [string]$FJson     = "F:\work\src\train\unified_jsons"
)

# shard dirs (exclude system subfolders)
$shards = Get-ChildItem -Directory $TrainRoot |
  Where-Object { @('unified_jsons','unified_aug','hetero_ready','logs') -notcontains $_.Name }

$names = $shards.Name
$cJson = Join-Path $TrainRoot "unified_jsons"
$jsonC = if(Test-Path $cJson){ Get-ChildItem $cJson -File -Filter *.json | % { $_.BaseName } } else { @() }
$jsonF = if(Test-Path $FJson){ Get-ChildItem $FJson -File -Filter *.json | % { $_.BaseName } } else { @() }

$have = ($jsonC + $jsonF) | Select-Object -Unique
$missing = $names | Where-Object { $have -notcontains $_ }

Write-Host ("Total shards={0}  JSON on C={1}  JSON on F={2}  Missing={3}" -f $names.Count, $jsonC.Count, $jsonF.Count, $missing.Count)
if($missing.Count -gt 0){
  "First 20 missing:", ($missing | Select-Object -First 20 -Join ", ")
}
