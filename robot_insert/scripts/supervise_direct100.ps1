param(
    [string]$JobDir = 'D:\robot_tools\aic_direct100_fresh_20261030',
    [string]$Distro = 'Ubuntu-24.04'
)
$ErrorActionPreference='Stop'
$mutex=[Threading.Mutex]::new($false,'Local\AICDirect100Fresh20261030')
if(-not $mutex.WaitOne(0)){throw 'The unattended AIC pipeline supervisor is already running.'}
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AICExecutionState {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint flags);
}
'@
New-Item -ItemType Directory -Path $JobDir -Force | Out-Null
$linuxJob='/mnt/d/robot_tools/aic_direct100_fresh_20261030'
$workspace='/home/skylerkuo/ros2_ws'
$events=Join-Path $JobDir 'windows_supervisor.jsonl'
function Record-Event($kind,$detail){
    @{time=(Get-Date).ToString('o');type=$kind;detail=$detail} |
        ConvertTo-Json -Compress -Depth 8 | Add-Content -LiteralPath $events -Encoding UTF8
}
function Read-State {
    try{return Get-Content -LiteralPath (Join-Path $JobDir 'state.json') -Raw | ConvertFrom-Json}
    catch{return $null}
}
try {
    [AICExecutionState]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    @{pid=$PID;started=(Get-Date).ToString('o');monitor_seconds=1200;keep_system_awake=$true;distro=$Distro} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $JobDir 'windows_supervisor.json') -Encoding UTF8
    for($attempt=1;$attempt -le 6;$attempt++){
        $state=Read-State
        if((Test-Path -LiteralPath (Join-Path $JobDir 'STOP')) -or $state.status -eq 'complete'){break}
        $stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
        $arguments=@('-d',$Distro,'--cd',$workspace,'--exec',"$workspace/.venv/bin/python",'-u',
            "$workspace/robot_insert/scripts/run_direct100_pipeline.py",'--job-dir',$linuxJob,'--monitor-seconds','1200')
        $process=Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $JobDir "pipeline_$stamp.log") `
            -RedirectStandardError (Join-Path $JobDir "pipeline_$stamp.err.log")
        Record-Event 'linux_pipeline_started' @{attempt=$attempt;windows_pid=$process.Id}
        $lastMonitor=Get-Date
        while(-not $process.HasExited){
            [AICExecutionState]::SetThreadExecutionState([uint32]2147483649) | Out-Null
            if(((Get-Date)-$lastMonitor).TotalSeconds -ge 1200){
                $state=Read-State
                Record-Event 'twenty_minute_monitor' $state
                $lastMonitor=Get-Date
            }
            Start-Sleep -Seconds 30
            $process.Refresh()
        }
        $process.WaitForExit()
        $state=Read-State
        Record-Event 'linux_pipeline_exited' @{attempt=$attempt;exit_code=$process.ExitCode;state=$state}
        if($state.status -in @('complete','stopped') -or (Test-Path -LiteralPath (Join-Path $JobDir 'STOP'))){break}
        if($attempt -eq 6){throw 'Pipeline recovery attempts exhausted; inspect events and stage logs.'}
        Start-Sleep -Seconds 30
    }
    Record-Event 'supervisor_finished' (Read-State)
} catch {
    Record-Event 'supervisor_error' $_.Exception.Message
    throw
} finally {
    [AICExecutionState]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    $mutex.ReleaseMutex();$mutex.Dispose()
}
