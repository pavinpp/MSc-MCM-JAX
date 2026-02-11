#!/bin/bash
# run_experiments.sh

echo "--- STARTING RESEARCH TOURNAMENT ---"

echo ">> [1/5] Running Baseline (Constant Pressure)..."
python main.py --mode run --strategy constant

echo ">> [2/5] Optimizing Square Wave (Pulse)..."
python main.py --mode optimize --strategy square

echo ">> [3/5] Optimizing Hammer Strategy (Shock)..."
python main.py --mode optimize --strategy hammer

echo ">> [4/5] Optimizing BioMimetic (Heartbeat)..."
python main.py --mode optimize --strategy biomimetic

echo ">> [5/5] Optimizing FreeForm AI (Discovery)..."
python main.py --mode optimize --strategy freeform

echo "--- TOURNAMENT COMPLETE. CHECK /plots AND /reports FOLDERS. ---"
