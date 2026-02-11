# run_experiments.ps1

Write-Output "--- STARTING RESEARCH TOURNAMENT ---"

Write-Output ">> [1/5] Running Baseline (Constant Pressure)..."
python main.py --mode run --strategy constant

Write-Output ">> [2/5] Optimizing Square Wave (Pulse)..."
python main.py --mode optimize --strategy square

Write-Output ">> [3/5] Optimizing Hammer Strategy (Shock)..."
python main.py --mode optimize --strategy hammer

Write-Output ">> [4/5] Optimizing BioMimetic (Heartbeat)..."
python main.py --mode optimize --strategy biomimetic

Write-Output ">> [5/5] Optimizing FreeForm AI (Discovery)..."
python main.py --mode optimize --strategy freeform

Write-Output "--- TOURNAMENT COMPLETE. CHECK /plots AND /reports FOLDERS. ---"
