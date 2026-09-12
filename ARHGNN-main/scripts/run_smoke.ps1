Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
python -m arhgnn.train --config configs/arhgnn.yaml --epochs 2 --bootstrap 10

