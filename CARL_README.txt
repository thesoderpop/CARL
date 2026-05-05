CARL package
============

Author and rights owner: Alexis Eleanor Fagan
Copyright (c) 2026 Alexis Eleanor Fagan. All rights reserved.

Files:
- CARL_neurips_style_paper.tex: self-contained NeurIPS-style LaTeX paper.
- carl_all_in_one.py: standalone CPU-native CARL implementation and stress test.
- carl_all_in_one_report.json: generated stress-test report from the script.
- CARL_neurips_style_paper.pdf: compiled PDF if pdflatex was available.

Run:
python carl_all_in_one.py --train_n 3000 --test_n 1200 --adversarial_n 600 --opaque_n 600 --out carl_all_in_one_report.json

Overleaf:
Upload CARL_neurips_style_paper.tex directly. It is standalone and does not require a separate neurips_2026.sty file.
For official NeurIPS submission, replace the standalone preamble with the official current-year NeurIPS style file and checklist.
