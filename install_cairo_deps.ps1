# Install Chocolatey (Windows package manager) if not already installed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    # Refresh PATH to include Chocolatey
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Install GTK+ and its dependencies
Write-Host "Installing GTK+ and dependencies..."
choco install -y cairo
choco install -y gtksharp
choco install -y gtk3

# Set environment variables
$cairoPath = "C:\Program Files\GTK3-Runtime Win64\bin"
if (-not ($env:Path -split ';' -contains $cairoPath)) {
    [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$cairoPath", [System.EnvironmentVariableTarget]::Machine)
    $env:Path += ";$cairoPath"
}

Write-Host "Installation complete! Please restart your terminal and IDE for changes to take effect."
