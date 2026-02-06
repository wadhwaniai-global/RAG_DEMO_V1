# PowerShell script to create bot user with all document filters

$USER_ID = "690390a6b3750101f0b11fda"
$TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjkwMzkwYTZiMzc1MDEwMWYwYjExZmRhIiwibmFtZSI6ImFzaGEiLCJ1c2VyX3R5cGUiOiJodW1hbiIsImV4cCI6MTc2NDQzODgzMn0.0wmZVYB-EVl8Sf48sj_fxGNiAuqMwx4pUU2uDuTiIJM"

$headers = @{
    "Authorization" = "Bearer $TOKEN"
    "Content-Type" = "application/json"
}

$body = @{
    name = "Asha-bot"
    email = "asha@example.com"
    user_type = "bot"
    document_filter = @(
        "ASHA Incentives 2024-2025",
        "ASHA Activities Guide",
        "ASHA Incentives April 2024",
        "ASHA NCD Module",
        "ASHA Booklet 2022",
        "asha_v1",
        "ENT Care Training Manual for MPW",
        "Eye Care Training Manual for ASHA",
        "FAQ_on_Immunization_for_Health_Workers-English",
        "ASHA_Handbook-Mobilizing_for_Action_on_Violence_against_Women_English",
        "ASHA_Induction_Module_English",
        "book-no-1",
        "book-no-2",
        "book-no-3",
        "book-no-4",
        "book-no-5",
        "book-no-6",
        "book-no-7",
        "Reaching_The_Unreached_Brochure_for_ASHA"
    )
} | ConvertTo-Json -Depth 3

Write-Host "Creating bot user with all document filters..." -ForegroundColor Green

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/api/v1/users/" `
                                 -Method Post `
                                 -Headers $headers `
                                 -Body $body

    Write-Host "`nBot user created successfully!" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "Error creating bot user:" -ForegroundColor Red
    Write-Host $_.Exception.Message
}