# ChromaDB Collection Check Script
Write-Host "=== ChromaDB Collection Check ===" -ForegroundColor Cyan

# Get all collections
Write-Host "`nFetching collections..." -ForegroundColor Yellow
$collections = Invoke-WebRequest -Uri "http://localhost:8081/api/v1/collections" -UseBasicParsing | ConvertFrom-Json

if ($collections.Count -eq 0) {
    Write-Host "No collections found in ChromaDB" -ForegroundColor Red
} else {
    Write-Host "Found $($collections.Count) collection(s):" -ForegroundColor Green
    
    foreach ($collection in $collections) {
        Write-Host "`n  Collection: $($collection.name)" -ForegroundColor Cyan
        Write-Host "  ID: $($collection.id)" -ForegroundColor Gray
        
        # Get collection details
        $collectionUrl = "http://localhost:8081/api/v1/collections/$($collection.name)"
        $details = Invoke-WebRequest -Uri $collectionUrl -UseBasicParsing | ConvertFrom-Json
        
        Write-Host "  Metadata: $($details.metadata | ConvertTo-Json -Compress)" -ForegroundColor Gray
        
        # Count vectors in collection
        $countUrl = "http://localhost:8081/api/v1/collections/$($collection.name)/count"
        $count = Invoke-WebRequest -Uri $countUrl -UseBasicParsing | ConvertFrom-Json
        
        Write-Host "  Vector Count: $count" -ForegroundColor Green
    }
}

Write-Host "`n=== End Check ===" -ForegroundColor Cyan
