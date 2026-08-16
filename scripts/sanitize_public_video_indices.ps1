param(
    [Parameter(Mandatory = $false)]
    [string]$Root = "results/development_20260816"
)

$ErrorActionPreference = "Stop"
$resolvedRoot = Resolve-Path -LiteralPath $Root

Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter "*VIDEO_INDEX.csv" |
    ForEach-Object {
        $rows = Import-Csv -LiteralPath $_.FullName
        foreach ($row in $rows) {
            if ($row.video_path) {
                $row.video_path = "not_committed/" + [IO.Path]::GetFileName($row.video_path)
            }
        }
        $rows | Export-Csv -LiteralPath $_.FullName -NoTypeInformation -Encoding UTF8
    }

Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter "*VIDEO_INDEX.json" |
    ForEach-Object {
        $records = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($record in $records) {
            if ($record.video_path) {
                $record.video_path = "not_committed/" + [IO.Path]::GetFileName($record.video_path)
            }
        }
        $records | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $_.FullName -Encoding UTF8
    }

Write-Output "Sanitised video indexes under $resolvedRoot"
