# Stitch MCP Launcher
# Use --transport http to skip OAuth discovery and connect directly
$apiKey = $env:STITCH_API_KEY
if ([string]::IsNullOrEmpty($apiKey)) {
    Write-Host "Error: STITCH_API_KEY environment variable is not set." -ForegroundColor Red
    exit 1
}
& "D:\Program Files\nodejs\npx.cmd" -y mcp-remote@0.1.16 https://stitch.googleapis.com/mcp --header "X-Goog-Api-Key:$apiKey" --transport sse-only
