# IsoSorb
IsoSorb is a Python-based scientific software for nonlinear adsorption isotherm modeling, multi-criteria statistical evaluation, and automatic model selection.

The software integrates adsorption isotherm fitting, statistical error analysis, residual diagnostics, and physicochemical interpretation into a unified computational framework designed for adsorption studies in environmental engineering, materials science, and chemical engineering.



# Features
* Nonlinear fitting of adsorption isotherm models
* Multi-model comparison
* Multi-criteria statistical evaluation
* Automatic model selection using SNE
* Residual analysis and diagnostic plots
* Automated physicochemical interpretation
* Graphical User Interface (GUI)
* Export of figures and statistical results
* Reproducible computational workflow


# Implemented Isotherm Models
IsoSorb currently includes the following adsorption isotherm models:
* Henry
* Langmuir
* Freundlich
* Temkin
* Dubinin–Radushkevich
* Redlich–Peterson
* Sips
* Toth
* Dubinin–Astakhov
* Volmer
* BET
* Aranovich

# Statistical Metrics
The software integrates multiple statistical functions for robust model evaluation:
* Coefficient of determination (R²)
* Sum of squared errors (SSE)
* Sum of absolute errors (SAE)
* Root mean square error (RMSE)
* Average relative error (ARE)
* Hybrid fractional error function (HYBRID)
* Marquardt’s percent standard deviation (MPSD)
* Chi-square (χ²)
* Sum of normalized errors (SNE)


# Software Architecture
IsoSorb follows a modular architecture composed of:
1. Data input module
2. Nonlinear fitting engine
3. Statistical evaluation module
4. Automatic model selection system
5. Residual analysis module
6. Physicochemical interpretation module
7. Visualization module
8. Export and reporting module

This modular design facilitates scalability, reproducibility, and future implementation of additional adsorption models and statistical approaches.


# Installation

## Clone the repository

```bash
git clone https://github.com/dagoarcecord/IsoSorb.git  
```

## Enter the project directory

```bash
cd IsoSorb


## Install dependencies
```bash
pip install -r requirements.txt
```


# Dependencies
IsoSorb was developed using Python 3.11.
Required libraries:
* NumPy
* SciPy
* pandas
* Matplotlib
* PyQt5


# Running the Software
Execute the main file:
```bash
IsoSorb.py
```


# Operating System
Validated under:
* Windows 10
* Windows 11


# Example Workflow
1. Import experimental equilibrium data (Ce, qe)
2. Select adsorption models
3. Perform nonlinear fitting
4. Evaluate statistical metrics
5. Identify the optimal model using SNE
6. Analyze residual distributions
7. Generate plots and export results


# Scientific Applications
IsoSorb can be applied to:
* Adsorption studies
* Water treatment processes
* Emerging contaminant removal
* Activated carbon characterization
* Bioadsorbent evaluation
* Surface interaction analysis
* Environmental remediation research


# Citation
If you use IsoSorb in your research, please cite:
Arce-Córdova, D., Gutiérrez Segura, E., Gutiérrez Hernández, R. F., Moreno Marcelino, J. E., Valle Mora, J. F., Maldonado, J. L., Ulloa Gutierrez, D. A., Rodiles-Cruz, N. C., & Hidalgo-López, R. (2026). IsoSorb: A Python-based tool for nonlinear adsorption isotherm modeling and multi-criteria model selection (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20274649
```

# License

This project is licensed under the GNU General Public License v3.0 


# Contact
For questions, bug reports, or collaboration:
rub.gutierrez@tapachula.tecnm.mx
