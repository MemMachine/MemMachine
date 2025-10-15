param([string]$Path = ".env")

Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#')) { return }

    if ($line -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') {
        $key = $matches[1].Trim()
        $val = $matches[2].Trim()
        if ($val -match '^"(.*)"$') { $val = $matches[1] }
        elseif ($val -match "^'(.*)'$") { $val = $matches[1] }
        Set-Item -Path ("Env:{0}" -f $key) -Value $val
    }
}