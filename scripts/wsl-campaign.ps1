# Validate each town in a fresh, process-owned CARLA server.
[CmdletBinding()]
param(
    [string]$CarlaRoot = 'C:\dev\carla\CARLA_0.9.16',
    [string]$Distro = 'Ubuntu',
    [string]$Python = '.cache/autopass-linux/bin/python',
    [string[]]$Towns = @('Town04','Town01','Town03','Town05'),
    [string[]]$Policies = @('autopass'),
    [string]$Scenarios = 'all',
    [string]$Out = 'runs/carla',
    [int]$Port = 25000,
    [int]$Seed = 0,
    [double]$MinFollow = 2.5,
    [ValidateSet('full','none')]
    [string]$Video = 'none',
    [ValidateSet('segdepth','rgbd')]
    [string]$Perception = 'segdepth',
    [ValidateSet('waypoint','imitation')]
    [string]$Steering = 'waypoint',
    [string]$Model = '',
    [ValidateSet('rule','llm_plan','llm_react')]
    [string]$Planner = 'rule'
)
$ErrorActionPreference = 'Stop'
$carlaExe = Join-Path $CarlaRoot 'CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe'
if (-not (Test-Path -LiteralPath $carlaExe -PathType Leaf)) { throw "CARLA binary absent: $carlaExe" }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is in use."
}
$work = (& wsl -d $Distro -e wslpath -a (Split-Path $PSScriptRoot -Parent)).Trim()
$gateway = (& wsl -d $Distro -e ip route show default).Split(' ', [StringSplitOptions]::RemoveEmptyEntries)[2]
$linuxUserHome = (& wsl -d $Distro -e printenv HOME).Trim()
$pythonBin = if ($Python.StartsWith('/')) { $Python } else { "$linuxUserHome/$Python" }
$linuxModel = if ($Model) { (& wsl -d $Distro -e wslpath -a $Model).Trim() } else { '' }
$errors = 0
foreach ($town in $Towns) {
    $ownedServer = Start-Process -FilePath $carlaExe -WorkingDirectory $CarlaRoot -WindowStyle Hidden -PassThru -ArgumentList @(
        'CarlaUE4','-RenderOffScreen','-quality-level=Low','-nosound',"-carla-rpc-port=$Port"
    )
    try {
        $until = [datetime]::UtcNow.AddSeconds(60)
        while (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
            if ($ownedServer.HasExited -or [datetime]::UtcNow -gt $until) { throw 'CARLA did not open its RPC port.' }
            Start-Sleep -Seconds 1
        }
        foreach ($policy in $Policies) {
            & wsl -d $Distro --cd $work -e env "CARLA_HOST=$gateway" "CARLA_PORT=$Port" "AUTOPASS_NANODET_MODEL=$linuxModel" $pythonBin -u -m autopass.carla_backend.run --load-town --town $town --scenarios $Scenarios --policy $policy --video $Video --out $Out --seed $Seed --perception $Perception --planner $Planner --steering $Steering --min-follow $MinFollow
            if ($LASTEXITCODE -ne 0) {
                $errors++
                Write-Warning "$town $policy exited $LASTEXITCODE. See run output."
            }
        }
    } finally {
        if (-not $ownedServer.HasExited) {
            $ownedServer.Kill()
            $ownedServer.WaitForExit(10000) | Out-Null
        }
    }
}

if ($errors) { throw "$errors campaign jobs failed." }
