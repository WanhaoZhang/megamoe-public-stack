param(
    [string]$ProjectRoot = 'D:\megamoe_train\moonep-benchmark-alignment-20261112\ep8-three-layer-comparison-20261113',
    [string]$RemoteRoot = '/home/z00956592/megaflux/planner-executor-ep8-lab'
)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath($ProjectRoot)
$index = Get-Content -LiteralPath (Join-Path $project 'archives\archive-index.json') -Raw | ConvertFrom-Json
if ($index.qualified_runs -ne 54) { throw 'Final archive does not cover54 runs' }
$members = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::Ordinal)
foreach ($archive in $index.archives) {
    foreach ($item in $archive.members) {
        if ($members.ContainsKey($item.member) -and $members[$item.member] -ne $item.sha256) { throw ('Conflicting archive member ' + $item.member) }
        $members[$item.member] = $item.sha256
    }
}
$freeze = Get-Content -LiteralPath (Join-Path $project 'evidence\workload-freeze.json') -Raw | ConvertFrom-Json
$frozenCount = 0
foreach ($group in @('sources','binaries')) {
    foreach ($entry in $freeze.$group.PSObject.Properties) {
        $member = if ($entry.Name.StartsWith($RemoteRoot + '/')) { $entry.Name.Substring($RemoteRoot.Length + 1) } else { 'external/' + $entry.Name.TrimStart([char]'/') }
        if (-not $members.ContainsKey($member) -or $members[$member] -ne $entry.Value) { throw ('Missing or changed frozen runtime ' + $entry.Name) }
        $frozenCount++
    }
}
$logs = $index.archives | Where-Object { $_.file -eq 'evidence-logs-and-measurements.tar.gz' }
$checked = 0
foreach ($entry in $logs.members) {
    $file = Join-Path $project ([string]$entry.member)
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw ('Missing extracted evidence ' + $entry.member) }
    if ((Get-Item -LiteralPath $file).Length -ne $entry.bytes) { throw ('Evidence size mismatch ' + $entry.member) }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw ('Evidence SHA mismatch ' + $entry.member) }
    $checked++
}
$metadata = $index.archives | Where-Object { $_.file -eq 'experiment-tools-config-routes-analysis.tar.gz' }
$metadataChecked = 0
foreach ($entry in $metadata.members) {
    if ($entry.member -match '^(routes|tables|figures|configs|reports|analysis-snapshots)/' -or $entry.member -in @('ACCEPTANCE.json','analysis-environment.json')) {
        $file = Join-Path $project ([string]$entry.member)
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw ('Missing extracted metadata ' + $entry.member) }
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw ('Metadata SHA mismatch ' + $entry.member) }
        $metadataChecked++
    }
}
$summary = @(Import-Csv -LiteralPath (Join-Path $project 'tables\summary.csv'))
$runs = @(Import-Csv -LiteralPath (Join-Path $project 'tables\runs.csv'))
$steps = @(Import-Csv -LiteralPath (Join-Path $project 'tables\steps.csv'))
if ($summary.Count -ne 18 -or $runs.Count -ne 54 -or $steps.Count -ne 1350) { throw 'Unexpected table coverage' }
if (@($summary | Where-Object { [int]$_.repeats -ne 3 }).Count -ne 0) { throw 'A point lacks3 repeats' }
$acceptance = Get-Content -LiteralPath (Join-Path $project 'ACCEPTANCE.json') -Raw | ConvertFrom-Json
if (-not $acceptance.passed -or $acceptance.formal_runs -ne 54) { throw 'Remote acceptance is missing or failed' }
foreach ($name in @('absolute-step-time','normalized-step-time','speedup-vs-moonep','whole-card-hbm')) {
    foreach ($ext in @('png','svg','pdf')) {
        $file = Join-Path $project ('figures\' + $name + '.' + $ext)
        if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-Item -LiteralPath $file).Length -lt 1000) { throw ('Missing plot ' + $file) }
    }
}
$transfer = @(Get-Content -LiteralPath (Join-Path $project 'artifacts\archive-transfer-checks.json') -Raw | ConvertFrom-Json)
if ($transfer.Count -ne $index.archives.Count -or @($transfer | Where-Object { -not $_.verified }).Count -ne 0) { throw 'Archive transfer verification incomplete' }
$receipt = [ordered]@{passed=$true;time=[DateTime]::UtcNow.ToString('o');formal_runs=54;repeats_per_point=3;extracted_evidence_files_verified=$checked;extracted_metadata_files_verified=$metadataChecked;frozen_source_and_binary_files_covered=$frozenCount;archives_verified=$transfer.Count;project=$project;large_debug_tensors='retained losslessly in checksum-verified archives';linux_products='retained in checksum-verified runtime and external-library archives'}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $project 'LOCAL-DELIVERY-ACCEPTANCE.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 8
