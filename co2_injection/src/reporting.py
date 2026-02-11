def generate_config_report(cfg):
    report = (
        "============================================================\n"
        "       APPENDIX A: SIMULATION CONFIGURATION\n"
        "============================================================\n"
        "1. Domain Geometry\n"
        f"   - Grid Dimensions:        {cfg.NX} x {cfg.NY}\n"
        f"   - Simulation Duration:    {cfg.STEPS} steps\n"
        "2. Fluid Parameters\n"
        f"   - Brine Density:          {cfg.RHO_BRINE:.2f}\n"
        f"   - CO2 Init Density:       {cfg.RHO_CO2:.2f}\n"
        f"   - Viscosity (Tau CO2):    {cfg.TAU_CO2:.2f}\n"
        "3. Reactive Transport\n"
        f"   - Salt Diffusion (D):     {cfg.D_SALT:.3f}\n"
        f"   - Solubility Limit (Ksp): {cfg.K_SP:.2f}\n"
        "============================================================\n"
    )
    return report


def generate_result_report(strategy, final_sat, baseline_sat, params):
    if baseline_sat is not None and baseline_sat > 0:
        gain = ((final_sat - baseline_sat) / baseline_sat) * 100.0
        gain_str = f"+{gain:.2f}%"
    else:
        gain_str = "N/A"
        baseline_sat = 0.0

    if "schedule" in params:
        param_str = "AI-Vector (Hidden)"
    else:
        param_str = ", ".join(
            [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in params.items()]
        )

    report = (
        "============================================================\n"
        "       APPENDIX B: OPTIMIZED RESULTS SUMMARY\n"
        "============================================================\n"
        "STRATEGY             | FINAL SAT    | GAIN       | TYPE      \n"
        "------------------------------------------------------------\n"
        f"Baseline             | {baseline_sat:.4f}       | +0.00%    | Static\n"
        f"{strategy:<20} | {final_sat:.4f}       | {gain_str:<9} | {type(params).__name__}\n"
        "------------------------------------------------------------\n"
        f"PARAMETERS: {param_str}\n"
        "============================================================\n"
    )
    return report


def save_report(text, filename):
    with open(filename, "w") as f:
        f.write(text)
    print(f"Report saved to {filename}")
