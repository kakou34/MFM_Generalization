"""Run all analysis_v2 scripts in order (outputs in <repo>/local/output_v2/).
    python run_all.py                 # statistics + figures
    python run_all.py --skip-features # skip t-SNE / W2 (slow)
"""
import subprocess
import sys

scripts = ["01_main_lmm.py", "02_covariate_adjustment.py", "03_mantel_test.py", "04_combat_comparison.py",
           "05_pooling_ablation.py"]
if "--skip-features" not in sys.argv:
    scripts.append("06_feature_space_figures.py")
for s in scripts:
    print(f"\n===== {s}")
    subprocess.run([sys.executable, s], check=True)
