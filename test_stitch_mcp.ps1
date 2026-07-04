$token = (gcloud auth application-default print-access-token).Trim()
Write-Host "Token fetched: $($token.Substring(0, 20))..."

$body = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

try {
    $response = Invoke-RestMethod `
        -Uri "https://stitch.googleapis.com/mcp" `
        -Method POST `
        -Headers @{
            "Authorization" = "Bearer $token"
            "Content-Type"  = "application/json"
        } `
        -Body $body `
        -TimeoutSec 15
    Write-Host "SUCCESS:" ($response | ConvertTo-Json -Depth 5)
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        Write-Host "Body: $($reader.ReadToEnd())"
    }
}
