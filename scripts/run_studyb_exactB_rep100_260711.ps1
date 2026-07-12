param(
  [int]$MaxWorkers = 2,
  [int]$Reps = 100,
  [string]$Methods = 'D-L,D-GL,D-AGL,E-L,E-CGL,E-ACGL',
  [string]$CellKeys = '',
  [bool]$SkipCompleted = $true,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rscript = 'C:\Program Files\R\R-4.2.1\bin\x64\Rscript.exe'
$runner = 'r\simulation\paper_eta_studyb_confirmatory_v2_260711.r'
$validMethods = @('D-L', 'D-GL', 'D-AGL', 'E-L', 'E-CGL', 'E-ACGL')
$methodValues = @($Methods.Split(',') | ForEach-Object { $_.Trim() })
if ($methodValues.Count -eq 0 -or ($methodValues | Where-Object { $_ -notin $validMethods })) {
  throw "Methods must be a comma-separated subset of: $($validMethods -join ',')."
}
$Methods = $methodValues -join ','
$methodTag = if ($Methods -eq ($validMethods -join ',')) {
  'all6'
} else {
  'subset_' + (($methodValues -join '_') -replace '[^A-Za-z0-9_-]', '')
}

$ebValues = @(
  [pscustomobject]@{ Tag = 'eb025'; Value = '0.025' },
  [pscustomobject]@{ Tag = 'eb05'; Value = '0.05' },
  [pscustomobject]@{ Tag = 'eb10'; Value = '0.10' }
)
$kappaPatterns = @(
  [pscustomobject]@{
    Tag = 'equal'; Label = 'equal'; Values = '45,45,45,45'
  },
  [pscustomobject]@{
    Tag = 'hetero'; Label = 'heterogeneous'; Values = '30,40,50,60'
  }
)
$cells = foreach ($eb in $ebValues) {
  foreach ($n in @(300, 1000)) {
    foreach ($kappa in $kappaPatterns) {
      [pscustomobject]@{
        Key = "$($kappa.Tag)_$($eb.Tag)_n$n"
        EbTag = $eb.Tag
        Eb = $eb.Value
        N = $n
        KappaTag = $kappa.Tag
        KappaLabel = $kappa.Label
        KappaValues = $kappa.Values
      }
    }
  }
}
if ($CellKeys.Trim()) {
  $requestedKeys = @($CellKeys.Split(',') | ForEach-Object { $_.Trim() })
  $unknownKeys = @($requestedKeys | Where-Object { $_ -notin $cells.Key })
  if ($unknownKeys.Count) {
    throw "Unknown CellKeys: $($unknownKeys -join ',')."
  }
  $cells = @($cells | Where-Object { $_.Key -in $requestedKeys })
}

if ($DryRun) {
  $cells | Select-Object EbTag, Eb, N, KappaTag, KappaLabel, KappaValues |
    Format-Table -AutoSize
  Write-Host "Dry run: cells=$($cells.Count), reps=$Reps, workers=$MaxWorkers, methods=$Methods"
  return
}

$running = @()
$completed = @()

function Collect-Completed {
  param([switch]$WaitForOne)
  do {
    $stillRunning = @()
    $found = $false
    foreach ($job in $script:running) {
      $job.Process.Refresh()
      if ($job.Process.HasExited) {
        $job.Process.WaitForExit()
        $exitCode = $job.Process.ExitCode
        if ($null -eq $exitCode) {
          $cellStatus = Join-Path $job.OutputDir "$($job.Label)_status.csv"
          $completionMarker = Join-Path $job.OutputDir "$($job.Label)_complete.ok"
          if ((Test-Path $cellStatus) -and (Test-Path $completionMarker)) {
            $terminal = @(Import-Csv $cellStatus)
            if ($terminal.Count -eq 1 -and $terminal[0].status -eq 'complete') {
              $exitCode = 0
            }
          }
          if ($null -eq $exitCode) { $exitCode = 1 }
        }
        $found = $true
        $script:completed += [pscustomobject]@{
          Label = $job.Label
          ExitCode = $exitCode
          OutputDir = $job.OutputDir
          FinishedAt = (Get-Date).ToString('s')
          Skipped = $false
        }
        Write-Host "[$($job.Label)] exit=$exitCode"
      } else {
        $stillRunning += $job
      }
    }
    $script:running = $stillRunning
    if ($WaitForOne -and -not $found) { Start-Sleep -Seconds 15 }
  } while ($WaitForOne -and -not $found)
}

foreach ($cell in $cells) {
  while ($running.Count -ge $MaxWorkers) { Collect-Completed -WaitForOne }

  $label = "paper_eta_studyb_v2_refitB_guard40_${methodTag}_$($cell.KappaTag)_$($cell.EbTag)_n$($cell.N)_rep${Reps}_path240_260712"
  $outDir = Join-Path $repo "results\$label"
  $cellStatus = Join-Path $outDir "${label}_status.csv"
  $completionMarker = Join-Path $outDir "${label}_complete.ok"
  if ($SkipCompleted -and (Test-Path $cellStatus) -and (Test-Path $completionMarker)) {
    $prior = @(Import-Csv $cellStatus)
    $isComplete = $prior.Count -eq 1 -and $prior[0].status -eq 'complete' -and
      [int]$prior[0].completed_reps -ge $Reps -and
      [int]$prior[0].expected_rows -eq [int]$prior[0].actual_rows -and
      [int]$prior[0].error_rows -eq 0
    if ($isComplete) {
      $completed += [pscustomobject]@{
        Label = $label
        ExitCode = 0
        OutputDir = $outDir
        FinishedAt = (Get-Date).ToString('s')
        Skipped = $true
      }
      Write-Host "[$label] already complete; skipped"
      continue
    }
  }
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null

  $env:USE_RCPP_HELPERS = '1'
  $env:V2_RUN_LABEL = $label
  $env:V2_OUT_DIR = "results/$label"
  $env:V2_N_VALUES = [string]$cell.N
  $env:V2_EB_VALUES = $cell.Eb
  $env:V2_KAPPA = $cell.KappaValues
  $env:V2_KAPPA_LABEL = $cell.KappaLabel
  $env:V2_METHODS = $Methods
  $env:V2_N_REP = [string]$Reps
  $env:V2_D = '200'
  $env:V2_NSTART = '10'
  $env:V2_MAX_ITER = '100'
  $env:V2_D_L_STEPS = '240'
  $env:V2_GROUP_STEPS = '240'
  $env:V2_ETA_STEPS = '240'
  $env:V2_REFIT_MAX_ITER = '160'
  $env:V2_REFIT_RETRY_MAX_ITER = '840'
  $env:V2_OPTIM_MAXIT = '80'
  $env:V2_REFIT_SHORTLIST = '40'
  $env:V2_REFIT_GUARD_RANK = '38'
  $env:V2_BASE_SEED = '20260711'
  $env:V2_CALIBRATION_ITER = '18'
  $env:V2_CALIBRATION_MC_N = '10000'
  $env:V2_VALIDATION_MC_N = '50000'
  $env:V2_TEST_N = '2000'
  $env:ORACLE_PILOT_SELECT_IC = 'BIC'
  $env:ORACLE_PILOT_ETA_REFIT_MODE = 'BIC_AFTER_EXACT'
  $env:ORACLE_PILOT_ETA_REFIT_SHORTLIST = '0'
  $env:ORACLE_PILOT_EXACT_REFIT_MAX_ITER = '160'
  $env:PAPER_S1_ETA_REFIT_MODE = 'BIC_AFTER_EXACT'
  $env:PAPER_S1_ETA_REFIT_SHORTLIST = '0'
  $env:PAPER_S1_EXACT_REFIT_MAX_ITER = '160'
  $env:PAPER_S1_MIN_REL_LAMBDA = '1e-3'
  $env:PAPER_S1_ADAPTIVE_GAMMA = '1'
  $env:PAPER_S1_ADAPTIVE_EPS = '1e-6'

  $stdout = Join-Path $outDir 'run.out'
  $stderr = Join-Path $outDir 'run.err'
  $process = Start-Process -FilePath $rscript -ArgumentList @($runner) `
    -WorkingDirectory $repo -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  $running += [pscustomobject]@{
    Label = $label
    OutputDir = $outDir
    Process = $process
  }
  Write-Host "[$label] started pid=$($process.Id)"
}

while ($running.Count) { Collect-Completed -WaitForOne }

$statusTag = if ($CellKeys.Trim()) { '_selected_retry' } else { '' }
$statusPath = Join-Path $repo "results\paper_eta_studyb_v2_refitB_guard40_${methodTag}_rep${Reps}${statusTag}_260712_status.csv"
$completed | Sort-Object Label | Export-Csv -Path $statusPath -NoTypeInformation
if (($completed | Where-Object ExitCode -ne 0).Count -gt 0) {
  throw "At least one Study B exact-B process failed. See $statusPath."
}
Write-Host "All Study B v2 exact-B shortlist cells completed. Status: $statusPath"
