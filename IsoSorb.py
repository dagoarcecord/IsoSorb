import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPalette, QColor, QTextDocument
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtGui




# Matplotlib 
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# SciPy para ajuste no lineal
from scipy.optimize import curve_fit

# cuadro de dialogo del selector
from PyQt5.QtWidgets import (
    QCheckBox, QVBoxLayout, QScrollArea, QPushButton, QLabel, QWidget,
    QTableWidget, QHeaderView, QTableWidgetItem, QTextBrowser, QHBoxLayout
)

# para exportar a  PDF 
from PyQt5.QtPrintSupport import QPrinter

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")  
    return os.path.join(base_path, relative_path)

# ==================================================
# MODELOS DE ISOTERMAS (no lineales)
# ==================================================

def langmuir(C, qmax, KL):
    return (qmax * KL * C) / (1.0 + KL * C)

def freundlich(C, KF, n):
    return KF * np.power(C, 1.0 / n)

def temkin(C, A, B):
    arg = A * C
    arg = np.where(arg <= 0, 1e-12, arg)
    return B * np.log(arg)

def dubinin_radushkevich(C, qDR, beta):
    eps = np.log(1.0 + 1.0 / (C + 1e-12))
    return qDR * np.exp(-beta * eps * eps)

def modelo_lineal(Ce, K):
    return K * Ce

def modelo_redlich_peterson(Ce, KR, aR, bR):
    denom = 1.0 + aR * np.power(Ce, bR)
    denom = np.where(denom == 0, 1e-12, denom)
    return (KR * Ce) / denom

def modelo_sips(Ce, qmax, KS, nS):
    x = np.power(KS * Ce, 1.0 / nS)
    return qmax * x / (1.0 + x)

def modelo_toth(Ce, qmax, KT, tT):
    denom = np.power(1.0 + np.power(KT * Ce, tT), 1.0 / tT)
    denom = np.where(denom == 0, 1e-12, denom)
    return (qmax * KT * Ce) / denom

def modelo_dubinin_astakhov(Ce, qmax, KD, nD):
    eps = np.log(1.0 + 1.0 / (Ce + 1e-12))
    val = - (eps / (KD + 1e-12)) ** nD
    return qmax * np.exp(val)

def modelo_volmer(Ce, qmax, KV):
    denom = 1.0 + KV * Ce
    denom = np.where(denom == 0, 1e-12, denom)
    return qmax * (KV * Ce) / denom

def modelo_BET(Ce, qm, CBET):
    denom = (1.0 - Ce) * (1.0 + (CBET - 1.0) * Ce)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return (qm * CBET * Ce) / denom

def modelo_aranovich(Ce, qmax, KA, aA):
    denom = np.power(1.0 + KA * Ce, aA)
    denom = np.where(denom == 0, 1e-12, denom)
    return (qmax * (KA * Ce)) / denom


# ==================================================
# FUNCIONES ERROR -- Como criterio de selección de ajuste del modelo de isoterma
# ==================================================
def compute_r2(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)

def compute_statistics(y_obs, y_pred, n_params):

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    resid = y_obs - y_pred

    n = len(y_obs)
    N = n
    p = n_params

    # ---- Función de suma de errores al cuadrado (SSE) --------
    SSE = np.sum(resid**2)

    # ----Función Suma de errores absolutos (SAE) ------
    SAE = np.sum(np.abs(resid))

    # ----Función  RMSE------
    RMSE = np.sqrt(SSE/n)

    # ----Función Error relativo promedio (ARE) ------
    ARE = 100/n * np.sum(np.abs(resid / (y_obs + 1e-12)))

    # ----Función  Chi-cuadrado -----
    X2 = np.sum((resid**2) / (y_obs + 1e-12))

    # ----Función de error fraccionario híbrido (HYBRID) -------
    if (N - p) > 0:
        HYBRID = 100/(N - p) * np.sum((resid**2) / (y_obs + 1e-12))
    else:
        HYBRID = np.nan

    # ----Función Desviación estándar porcentual de Marquardt (MPSD) -----
    if (N - p) > 0:
        MPSD = 100 * np.sqrt((1/(N - p)) * np.sum(((resid)/(y_obs + 1e-12))**2))
    else:
        MPSD = np.nan

    # ----Función Coeficiente de determinación (R2) ------
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    R2 = 1 - SSE/ss_tot if ss_tot != 0 else 0

    return {
        "R2": R2,
        "SSE": SSE,
        "SAE": SAE,
        "ARE": ARE,
        "X2": X2,
        "HYBRID": HYBRID,
        "MPSD": MPSD,
        "RMSE": RMSE
    }
def compute_sne(results):

    error_names = ["SSE","SAE","ARE","X2","HYBRID","MPSD"]

    # encontrar máximos
    max_values = {}

    for err in error_names:
        max_values[err] = max(r["stats"][err] for r in results)

    # ---- Función Suma de errores normalizados (SNE) -----
    for r in results:

        sne = 0

        for err in error_names:

            val = r["stats"][err]

            max_val = max_values[err]

            if max_val == 0:
                ne = 0
            else:
                ne = val / max_val

            sne += ne

        r["stats"]["SNE"] = sne

    return results

def safe_curve_fit(func, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), maxfev=10000):
    try:
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds, maxfev=maxfev)
        return popt, pcov, None
    except Exception as e:
        return None, None, str(e)

# ==================================================
# MAPA DE MODELOS ISOTERMAS
# ==================================================
MODELOS_ISOTERMAS = {
    "Linear": (modelo_lineal, [0.1], ( -np.inf, np.inf), ["K [L/g]"]),
    "Langmuir": (langmuir, [1.0, 0.1], (0, np.inf), ["qmax [mg/g]", "KL [L/mg]"]),
    "Freundlich": (freundlich, [1.0, 1.5], (0, np.inf), ["KF [(mg/g)/(mg/L)^(1/n)]", "n [-]"]),
    "Temkin": (temkin, [1.0, 1.0], (-np.inf, np.inf), ["A [-]", "B [mg/g]"]),
    "Dubinin-Radushkevich": (dubinin_radushkevich, [1.0, 0.1], (0, np.inf), ["qDR [mg/g]", "beta []"]),
    "Redlich–Peterson": (modelo_redlich_peterson, [1.0, 0.1, 0.5], (-np.inf, np.inf), ["KR", "aR", "bR"]),
    "Sips": (modelo_sips, [1.0, 0.1, 1.0], (0, np.inf), ["qmax [mg/g]", "KS [L/mg]", "nS [-]"]),
    "Toth": (modelo_toth, [1.0, 0.1, 1.0], (0, np.inf), ["qmax [mg/g]", "KT [L/mg]", "tT [-]"]),
    "Dubinin–Astakhov": (modelo_dubinin_astakhov, [1.0, 1.0, 1.0], (0, np.inf), ["qmax [mg/g]", "KD []", "nD [-]"]),
    "Volmer": (modelo_volmer, [1.0, 0.1], (0, np.inf), ["qmax [mg/g]", "KV [L/mg]"]),
    "BET": (modelo_BET, [1.0, 1.0], (-np.inf, np.inf), ["qm [mg/g]", "CBET [-]"]),
    "Aranovich": (modelo_aranovich, [1.0, 0.1, 1.0], (0, np.inf), ["qmax [mg/g]", "KA [L/mg]", "aA [-]"])
}


# ==================================================
# FUNDAMENTOS DE LOS MODELOS DE ISOTERMAS
# ==================================================

FUNDAMENTOS = {}
img_path = resource_path("assets/lineal.png")
FUNDAMENTOS["Linear"] = {
    "full": """
    <h1 style="color:#00E5C0;">Linear</h1>
    <p style="text-align: justify; font-size:16px;">
     The linear adsorption model describes the relationship between the adsorbed amount and the equilibrium concentration of the adsorbate as a direct proportionality, assuming a homogeneous surface with an abundance of active sites and no saturation effects or interactions between the adsorbed species. The distribution constant reflects the affinity between the adsorbate and the adsorbent, such that high values indicate greater adsorption capacity, while low values suggest weak interactions. This parameter is determined experimentally from equilibrium data by plotting the adsorbed amount against the equilibrium concentration, where the slope of the line corresponds to this constant. This model is primarily applicable to low concentrations, where the adsorbent surface is not near saturation.
    </p>
    <p style="text-align: center;">
    <img src="lineal.png" style="height: 60px;">
</p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>q<sub>e</sub></i> is the amount adsorbed at equilibrium (mg/g), <i>C<sub>e</sub></i> is the equilibrium concentration in the fluid phase (mg/L), and <i>K<sub>H</sub></i> is the distribution constant or adsorption coefficient (L/g).
    </p>

    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>K<sub>H</sub></i> (distribution constant)</b> represents the affinity between the adsorbate and the adsorbent.


<li><b><i>K<sub>H</sub></i> high</b> → strong adsorption (the solute “prefers” the solid)</li>
<li><b><i>K<sub>H</sub></i> low</b> → weak adsorption (the solute remains in solution)</li>
</ul>
</p>
         """
}

FUNDAMENTOS["Langmuir"] = {
    "full": """
    <h1 style="color:#00E5C0;">Langmuir</h1>
    <p style="text-align: justify; font-size:16px;">
     The Langmuir isotherm is one of the most widely used models in the study of adsorption and was originally developed to describe gas–solid systems (Langmuir, 1916, 1918). This model assumes that adsorption occurs as a monolayer on a surface with a finite number of active sites, which are well-defined, equivalent to one another, and possess the same energy. Furthermore, it assumes that there are no lateral interactions or steric effects between the adsorbed molecules, even when they are located at adjacent sites. From a theoretical standpoint, the model is based on the hypothesis of a homogeneous surface, where all adsorbed molecules exhibit constant values of enthalpy and activation energies during the sorption process, implying that each active site has the same affinity for the adsorbate.    </p>
    <p style="text-align: center;">
    <img src="langmuir.png" style="height: 60px;">
</p>
    
    <p style="text-align: justify; font-size:16px;">
    where <i>q<sub>e</sub></i>: amount adsorbed at equilibrium (mg/g), <i>q<sub>m</sub></i>: maximum adsorption capacity (mg/g), <i>C<sub>e</sub></i>: equilibrium concentration in the liquid phase (mg/L), <i>K<sub>L</sub></i>: Langmuir constant or affinity constant (L/mg)</p>

    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>q<sub>m</sub></i> (Maximum Capacity)</b> represents the maximum amount that the material can adsorb. It indicates the capacity of the monolayer and depends on the surface area and the number of active sites.


<ul style="font-size:16px;">
<li><b><i>q<sub>m</sub></i> high</b> → higher adsorption capacity</li>
<li><b><i>q<sub>m</sub></i> low</b> → lower adsorption capacity</li>
</ul>
</p>

<p style="font-size:16px;">
The <b><i>K<sub>L</sub> (Langmuir constant)</b> represents the affinity between the adsorbate and the adsorbent, related to the adsorption energy.
</p>

<ul style="font-size:16px;">
<li><b><i>K<sub>L</sub> high</b> → strong interaction (favorable adsorption)</li>
<li><b><i>K<sub>L</sub> low</b> → weak interaction</li>
</ul>

   """
}

FUNDAMENTOS["Freundlich"] = {
    "full": """
    
    <h1 style="color:#00E5C0;">Freundlich</h1>
    <p style="text-align: justify; font-size:16px;">
     The Freundlich model is a widely used empirical isotherm for describing adsorption processes on heterogeneous surfaces, where active sites have different adsorption energies. Unlike the Langmuir model, it does not assume a uniform surface or the formation of a monolayer, but rather allows for the representation of multilayer adsorption, making it more suitable for real-world systems.
     This model is based on a nonlinear relationship between the adsorbed amount and the equilibrium concentration, and is particularly useful in intermediate concentration ranges, where the coverage fraction can approach values close to 50%. In addition, it incorporates an adsorption intensity parameter, represented by n, which allows the favorability of the process to be evaluated.

    </p>
    <p style="text-align: center;">
    <img src="Freundlich.png" style="height: 60px;">
    </p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>q<sub>e</sub></i>: amount adsorbed at equilibrium (mg/g), <i>C<sub>e</sub></i>: equilibrium concentration in the liquid phase (mg/L), <i>K<sub>F</sub></i>: Freundlich constant (mg/g)(L/mg)^(1/n), <i>n</i>: adsorption intensity parameter (dimensionless)


    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
    The <b> <i>K<sub>F</sub></i> (Freundlich constant)</b> indicates the adsorption capacity of the system.

    <li><b><i>K<sub>F</sub></i> high</b> → higher adsorption capacity</li>
    <li><b><i>K<sub>F</sub></i> low</b> → lower adsorption capacity</li>
    </ul>
    </p>

     <p style="font-size:16px;">
    The <b><i>n</i> adsorption intensity </b> Describes the intensity or favorability of the process.

    <li><b> <i>n</i> > 1 </b> → Favorable adsorption.</li>
    <li><b> <i>n</i> &lt; 1 </b> → Unfavorable adsorption.</li>
    </ul>
    </p>
    
    """
    
}

FUNDAMENTOS["Temkin"] = {
    "full": """
<h1 style="color:#00E5C0;">Temkin</h1>
    <p style="text-align: justify; font-size:16px;">
     The Temkin model describes the adsorption process by considering the interactions between the adsorbed molecules and the adsorbent surface, assuming that the heat of adsorption decreases linearly with increasing surface coverage. Unlike other models, this approach recognizes that the adsorption energy is not constant but varies due to adsorbate–adsorbate effects, allowing for a more realistic representation in systems where such interactions are significant. Furthermore, the model assumes a uniform distribution of binding energies up to a maximum value, establishing a logarithmic relationship between the adsorbed amount and the equilibrium concentration.

    </p>
    <p style="text-align: center;">
    <img src="temkin.png" style="height: 60px;">
</p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>q<sub>e</sub></i>: amount adsorbed at equilibrium (mg/g), <i>C<sub>e</sub></i>: equilibrium concentration in the liquid phase (mg/L), <i>A</i>: Temkin equilibrium constant (L/g), <i>b</i>: constant related to the heat of adsorption (J/mol),<i>B</i> Temkin constant (mg/g),<i>R</i> gas constant (8.314 J/mol·K),<i>T</i>absolute temperature (K)


    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>A</i> (Temkin constant)</b> Related to the adsorbate–adsorbent affinity, it represents the equilibrium constant of the process.
</p>

<ul style="font-size:16px;">
<li><b><i>A</i> high</b> → higher affinity / favorable adsorption</li>
<li><b><i>A</i> low</b> → lower affinity</li>
</ul>

<p style="font-size:16px;">
The <b><i>b</i> constant related to the heat of adsorption:  </b> Associated with the variation in adsorption energy, it indicates how the heat of adsorption changes with coverage.
</p>

<ul style="font-size:16px;">
<li><b> <i>High b</i></b> → smaller energy variation.</li>
<li><b> <i>Low b</i></b> → greater decrease in heat of adsorption.</li>
</ul>    
    
    """
}

FUNDAMENTOS["Dubinin-Radushkevich"] = {
    "full": """
    <h1 style="color:#00E5C0;">Dubinin-Radushkevich</h1>
    <p style="text-align: justify; font-size:16px;">
     The Dubinin–Radushkevich (D–R) model is an empirical isotherm based on Polanyi’s potential theory, used to describe adsorption processes on heterogeneous surfaces and in systems with porous structures. This model assumes that adsorption occurs via a pore-filling mechanism and that the distribution of adsorption energies follows a Gaussian function. Unlike other models, it does not assume a homogeneous surface or a constant energy, which allows for its application in more complex systems.

    </p>
    <p style="text-align: center;">
    <img src="DR1.png" style="height: 60px;">
    </p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>q<sub>e</sub></i>: amount adsorbed at equilibrium (mg/g), <i>q<sub>mD-R</sub></i>: is the maximum amount adsorbed (mg/g), <i>K<sub>DR</sub></i>: is the D-R model constant (mol²/kJ²), <i>ϵ</i>: is the adsorption potential based on Polanyi's theory
    
    </p>
    <p style="text-align: center;">
    <img src="DR2.png" style="height: 60px;">
    </p>

     <p style="text-align: justify; font-size:16px;">
      where <i>C<sub>s</sub></i>: is the solubility of the adsorbate (mg/L), <i>C<sub>e</sub></i>: is the equilibrium concentration of the adsorbate (mg/g), A distinctive feature of the D-R model is that it allows the calculation of the average free energy of adsorption, E (kJ/mol), using:

    <p style="text-align: center;">
    <img src="DR3.png" style="height: 60px;">
    </p>          


    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>K<sub>DR</sub></i> (D-R model constant)</b> Related to the adsorption energy, it allows the average energy of the process to be calculated.
</p>

<ul style="font-size:16px;">
<li><b><i>K<sub>DR</sub></i> high</b> → lower average adsorption energy</li>
<li><b><i>K<sub>DR</sub></i> low</b> → higher adsorption energy</li>
</ul>

<p style="font-size:16px;">
The <b><i>E</i> average adsorption energy </b> indicates the nature of the process.
</p>

<ul style="font-size:16px;">
<li><b> <i>E</i> &lt;8 kJ/mol </b> → Physical adsorption.</li>
<li><b>  8 kJ/mol &lt;<i>E</i> &lt; 16 kJ/mol </b> → Chemical adsorption.</li>
</ul>
    """
}

FUNDAMENTOS["Redlich–Peterson"] = {
    "full": """
    <h1 style="color:#00E5C0;">Redlich–Peterson</h1>
    <p style="text-align: justify; font-size:16px;">
     The Redlich–Peterson model is a three-parameter hybrid isotherm that combines features of the Langmuir and Freundlich models, allowing it to describe adsorption processes on both homogeneous and heterogeneous surfaces. This model does not assume a specific ideal behavior, so it can be applied over a wide range of concentrations. Its flexibility lies in the exponent 
𝑔, which introduces an additional degree of adjustment to represent the heterogeneity of the system. Depending on the value of this parameter, the model can behave like Langmuir at high concentrations or like Freundlich under specific conditions, making it a useful tool for fitting complex experimental data.

    </p>
    <p style="text-align: center;">
    <img src="RP.png" style="height: 60px;">
    </p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>K<sub>RP</sub></i>: is the equilibrium constant (L/g), <i>α<sub>RP</sub></i>: is a constant related to the adsorption energy (L^g.m^g), <i>g</i>: is the heterogeneity exponent; depending on the value of <i>g</i>, the following cases may occur: <i>g</i> = 1 → Langmuir-type behavior, <i>g</i> &lt; 1 → Heterogeneous surface. The Redlich-Peterson model can be reduced as follows:
         
    <p style="text-align: justify; font-size:16px;">
      <i>g</i> = 1 →  Langmuir-type behavior
      
      </p>
      <p style="text-align: center;">
      <img src="RP1.png" style="height: 60px;">
      </p>
      
      <p style="text-align: justify; font-size:16px;">
        <i>C<sub>e</sub></i>  → ∝  Freundlich-type behavior
        
        </p>
        <p style="text-align: center;">
        <img src="RP2.png" style="height: 60px;">
        </p>
      
        <p style="text-align: justify; font-size:16px;">
          <i>C<sub>e</sub></i>  → 0  The equation reduces to the Linear model
          
          </p>
          <p style="text-align: center;">
          <img src="RP3.png" style="height: 60px;">
          </p> 
    

    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>K<sub>RP</sub></i> equilibrium constant, </b> related to the adsorbate’s affinity for the adsorbent.
</p>

<ul style="font-size:16px;">
<li><b><i>K<sub>RP</sub></i> high</b> → greater affinity and greater adsorption</li>
<li><b><i>K<sub>RP</sub></i> low</b> → lower affinity</li>
</ul>


   <p style="font-size:16px;">
The <b> <i>α<sub>RP</sub></i> (constant associated with the adsorption energy), </b> related to the curvature of the isotherm
</p>

<ul style="font-size:16px;">
<li><b><i>α<sub>RP</sub></i> high</b> → lower adsorption capacity at high concentrations</li>
<li><b><i>α<sub>RP</sub></i> low</b> → higher adsorption capacity</li>
</ul>   

    

    """
}

FUNDAMENTOS["Sips"] = {
    "full": """
    
    <h1 style="color:#00E5C0;">SIPS</h1>
    <p style="text-align: justify; font-size:16px;">
     The SIPS model is a three-parameter hybrid isotherm that combines the characteristics of the Langmuir and Freundlich models, allowing for the description of adsorption processes on both homogeneous and heterogeneous surfaces. This model introduces a heterogeneity exponent that corrects the limitation of the Langmuir model by accounting for variations in adsorption energy on the adsorbent surface. In its formulation, the model assumes a monolayer adsorption process at high concentrations, while at low concentrations it exhibits Freundlich-type behavior, making it suitable for real-world systems where energy distribution is non-uniform.

From a theoretical standpoint, the model can be derived from a dynamic equilibrium between adsorption and desorption rates, incorporating the concept of fractional surface coverage and considering that an adsorbate molecule can occupy a fraction of the available sites. This formulation allows for describing how adsorption energy is distributed over heterogeneous surfaces. Furthermore, the SIPS model avoids predicting infinite adsorption at high concentrations, unlike the Freundlich model, making it a more realistic tool for saturated systems.

    </p>
    <p style="text-align: center;">
    <img src="SIPS.png" style="height: 60px;">
    </p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>q<sub>ms</sub></i>: is the maximum adsorbed amount (mg/g), <i>K<sub>s</sub></i>: is the SIPS affinity constant (L^ns.mg^(-ns) ), <i>n<sub>s</sub></i>: is the heterogeneity constant; depending on the value of <i>n<sub>s</sub></i>, the following cases may occur:
         
    <p style="text-align: justify; font-size:16px;">
      <i>n<sub>s</sub></i> = 1 →  Langmuir-type behavior
      
      </p>
      <p style="text-align: center;">
      <img src="langmuir.png" style="height: 60px;">
      </p>
      
      <p style="text-align: justify; font-size:16px;">
        <i>C<sub>e</sub></i>  → 0  Freundlich-type behavior
        
        </p>
        <p style="text-align: center;">
        <img src="sips2.png" style="height: 60px;">
        </p>
      

    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>K<sub>s</sub></i> SIPS affinity constant, </b> related to the adsorption energy and adsorbate–adsorbent affinity.
</p>

<ul style="font-size:16px;">
<li><b><i>K<sub>s</sub></i> high</b> → higher affinity and greater adsorption</li>
<li><b><i>K<sub>s</sub></i> low</b> → lower affinity</li>
</ul>
     
    """

}

FUNDAMENTOS["Toth"] = {
    "full": """
    
    <h1 style="color:#00E5C0;">Toth</h1>
    <p style="text-align: justify; font-size:16px;">
     The Tóth model is an empirical isotherm developed as a modification of the Langmuir model, with the aim of improving the description of adsorption systems on heterogeneous surfaces, especially over a wide range of concentrations. This model introduces a heterogeneity parameter that corrects the overestimation of the Langmuir model at high concentrations, providing a more accurate fit in both the low- and high-coverage regions.

Based on a quasi-Gaussian distribution of adsorption energies, the Tóth model assumes that active sites have different affinities for the adsorbate, making it suitable for real-world systems. At low concentrations, the model tends to behave linearly, similar to Henry’s law, whereas when the heterogeneity parameter takes the value of one, the equation reduces to the Langmuir model. Therefore, the Tóth model can be interpreted as a flexible extension of Langmuir that incorporates the effects of energy heterogeneity without losing the ability to describe the saturation of the system.

    </p>
    <p style="text-align: center;">
    <img src="TOTH.png" style="height: 60px;">
    </p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>K<sub>T</sub></i>:  represents the maximum adsorption capacity of the adsorbent (mg/g), <i>α<sub>T</sub></i>: is the Toth constant related to affinity, which increases with temperature, <i>Z</i>: is a component that describes the heterogeneity of the system and is independent of temperature; depending on the value of <i>Z</i>, the following cases may occur:
         
    <p style="text-align: justify; font-size:16px;">
      <i>Z</i> = 1 →  Langmuir-type behavior
      
      </p>
      <p style="text-align: center;">
      <img src="langmuir.png" style="height: 60px;">
      </p>
      
      <p style="text-align: justify; font-size:16px;">
        <i>Z</i>  ≠ 1 at low concentrations assumes linear Henry behavior
        
        </p>
        <p style="text-align: center;">
        <img src="linear.png" style="height: 60px;">
        </p>

    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>K<sub>T</sub></i> (maximum adsorption capacity), </b> is related to the system's saturation limit.
</p>

<ul style="font-size:16px;">
<li><b><i>K<sub>T</sub></i> high</b> → greater adsorption capacity</li>
<li><b><i>K<sub>T</sub></i> low</b> → lower adsorption capacity</li>
</ul>

<p style="text-align: justify;color:#00E5C0; font-size:20px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>α<sub>T</sub></i> Tóth constant, </b> related to the adsorbate–adsorbent affinity.
</p>

<ul style="font-size:16px;">
<li><b><i>α<sub>T</sub></i> high</b> → lower affinity</li>
<li><b><i>α<sub>T</sub></i> low</b> → higher affinity</li>
</ul>    
    """
    
}

FUNDAMENTOS["Dubinin–Astakhov"] = {
    "full": """
    <h1 style="color:#00E5C0;">Dubinin–Astakhov</h1>
    <p style="text-align: justify; font-size:16px;">
     The Dubinin–Astakhov (D–A) model is an extension of the Dubinin–Radushkevich model, developed from Polanyi’s potential theory, and is primarily used to describe adsorption processes in microporous materials. Unlike classical models such as Langmuir or Freundlich, the D–A model does not assume the formation of a monolayer on a homogeneous surface, but rather considers the mechanism of filling the volume of the micropores as the dominant adsorption process.

This model is based on the idea that adsorbate molecules are attracted to the micropores by a potential energy field, the magnitude of which is expressed by the adsorption potential. This potential represents the work required to transfer a molecule from the fluid phase into the interior of the pore. The Dubinin–Astakhov equation introduces an additional heterogeneity parameter that allows for a more accurate description of systems with non-uniform energy distributions, significantly improving the experimental fit compared to the Dubinin–Radushkevich model, especially in materials with complex microporosity.

    </p>
    <p style="text-align: center;">
    <img src="DA.png" style="height: 60px;">
    </p>
    
    The liquid-phase adsorption potential is
    
    </p>
    <p style="text-align: center;">
    <img src="DA2.png" style="height: 60px;">
    </p>
    
    <p style="text-align: justify; font-size:16px;">
     where <i>E<sub>DA</sub></i> is the characteristic adsorption energy, <i>η<sub>DA</sub></i> is the heterogeneity exponent, <i>C<sub>s</sub></i>is the saturation concentration (mg/L), <i>C<sub>e</sub></i> is the equilibrium concentration (mg/L), and <i>ε</i> is the adsorption potential (kJ/mol) 
         
    
    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>q<sub>mD-A</sub></i> is the maximum adsorption capacity, </b> the maximum amount of adsorbate that can fill the micropores of the adsorbent.
</p>

<ul style="font-size:16px;">
<li><b><i>q<sub>mD-A</sub></i> high</b> → Greater micropore volume - higher adsorption capacity</li>
<li><b><i>q<sub>mD-A</sub></i> low</b> → Smaller micropore volume - higher adsorption capacity</li>
</ul>

    
    <p style="font-size:16px;">
The <b> <i>E<sub>DA</sub></i> characteristic adsorption energy, </b> represents the average energy required for the adsorbate to be retained within the micropores of the adsorbent.
</p>

<ul style="font-size:16px;">
<li><b><i>E<sub>DA</sub></i> high</b> → Stronger interactions, more intense adsorption, and higher affinity</li>
<li><b><i>E<sub>DA</sub></i> low</b> → Weaker interactions, less intense adsorption, and lower affinity</li>
</ul>

 <p style="font-size:16px;">
The <b> <i>η<sub>DA</sub></i> heterogeneity parameter, </b> indicates how uniform or heterogeneous the adsorption energy is within the micropores. If all sites have similar energy, it corresponds to a homogeneous system; if there is a large variation in energies, it is a heterogeneous system
</p>

<ul style="font-size:16px;">
<li><b><i>η<sub>DA</sub></i> ≈2  (high) </b> → More uniform energy distribution, Dubinin–Radushkevich-type behavior</li>
<li><b><i>η<sub>DA</sub></i> &lt;2 (low)</b> → Large variation in adsorption energies, heterogeneous surface</li>
</ul>
    """
}

FUNDAMENTOS["Volmer"] = {
    "full": """
    <h1 style="color:#00E5C0;">Volmer</h1>
    <p style="text-align: justify; font-size:16px;">
     The Volmer isotherm model, originally proposed by Volmer (1925), is based on the assumption that adsorbed molecules are mobile on the surface of the adsorbent, unlike the Langmuir model, where molecules remain fixed. Furthermore, the model assumes that there are no lateral interactions between the adsorbed molecules, making it an intermediate representation between ideal surfaces and more complex systems.

This model is particularly relevant for describing adsorption processes in which the adsorbed phase exhibits a certain degree of freedom of movement, which introduces non-ideal behavior in the formation of the monolayer. From a physical standpoint, the model can be interpreted as an extension of the Langmuir model that incorporates entropic effects associated with surface mobility.

    </p>
    <p style="text-align: center;">
    <img src="VOLMER.png" style="height: 60px;">
    </p>
    
       
    <p style="text-align: justify; font-size:16px;">
     where <i>b<sub>V</sub></i> is the affinity constant (L/mg), <i>θ<sub>e</sub></i>=<i>q<sub>e</sub></i>⁄<i>q<sub>m</sub></i> is the coverage fraction (mg/L), and <i>C<sub>e</sub></i> is the equilibrium adsorbate concentration (mg/L).  
         
    
    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>b<sub>V</sub></i> (affinity constant of the Volmer model), </b> related to the adsorption energy and the probability of active site occupancy..
</p>

<ul style="font-size:16px;">
<li><b><i>b<sub>V</sub></i> high</b> → higher adsorbent–adsorbate affinity, favoring adsorption at low concentrations.</li>
<li><b><i>b<sub>V</sub></i> low</b> → lower affinity, requiring higher concentrations to achieve significant coverage</li>
</ul>

    
    <p style="font-size:16px;">
The <b> <i>q<sub>m</sub></i> (maximum adsorption capacity), </b> represents the amount of adsorbate needed to completely cover the adsorbent surface, forming a monolayer.
</p>

<ul style="font-size:16px;">
<li><b><i>q<sub>m</sub></i> high</b> → higher adsorption capacity</li>
<li><b><i>q<sub>m</sub></i> low</b> → lower adsorption capacity</li>
</ul>

 <p style="font-size:16px;">
The <b> <i>θ<sub>e</sub></i> (coverage fraction), </b> indicates the degree of surface occupancy
</p>

<ul style="font-size:16px;">
<li><b><i>θ<sub>e</sub></i> ≈0 </b> → low occupancy of active sites</li>
<li><b><iθ<sub>e</sub></i> ≈1 </b> → represents surface saturation</li>
</ul>
    """
}

FUNDAMENTOS["BET"] = {
    "full": """
    <h1 style="color:#00E5C0;">BET</h1>
    <p style="text-align: justify; font-size:16px;">
     The Brunauer–Emmett–Teller (BET) isotherm is an extension of the Langmuir model that allows for the description of multilayer adsorption on the surface of a solid. This model assumes that adsorption occurs sequentially, where molecules adsorbed in the first layer act as active sites for the formation of subsequent layers.

The model distinguishes between the adsorption energy of the first layer, associated with the adsorbate–adsorbent interaction, and the energy of the upper layers, which is considered constant and equal to the liquefaction energy of the adsorbate. It also assumes that there is no lateral migration between layers and that equilibrium is established when the adsorption and desorption rates are equal at each level.

Due to these assumptions, the BET model is widely used to characterize porous materials and estimate the specific surface area, being particularly applicable in physical adsorption systems in the gas phase and, to a lesser extent, in the liquid phase.

    </p>
    <p style="text-align: center;">
    <img src="BET1.png" style="height: 60px;">
    </p>
    
       
    <p style="text-align: justify; font-size:16px;"> 
    where <i>q<sub>e</sub></i> is the amount of adsorbate adsorbed at equilibrium (mg/g), <i>q<sub>mBET</sub></i> is the maximum adsorption capacity in the monolayer (mg/g), <i>K<sub>BET</sub></i> is the overall adsorption equilibrium constant (L/mg), and <i>C<sub>s</sub></i> is the solubility of the adsorbate in the solvent (mg/L).
    </p>
         
    
    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>q<sub>mBET</sub></i> (monolayer capacity), </b> is the amount of adsorbate required to form a complete monolayer.
</p>

<ul style="font-size:16px;">
<li><b><i>q<sub>mBET</sub></i> high</b> → larger surface area / higher capacity.</li>
<li><b><i>q<sub>mBET</sub></i> low</b> → smaller surface area</li>
</ul>

    
    <p style="font-size:16px;">
The <b> <i>K<sub>BET</sub></i> (global adsorption equilibrium constant), </b> represents the energy ratio between the first layer and subsequent layers.
</p>

<ul style="font-size:16px;">
<li><b><i>K<sub>BET</sub></i> &gt;1</b> → High adsorbent–adsorbate affinity, strong interaction in the first layer</li>
<li><b><i>K<sub>BET</sub></i> ≈1</b> → Similar energy between layers, more homogeneous adsorption</li>
<li><b><i>K<sub>BET</sub></i> &lt;1</b> → Unfavorable adsorption</li>
</ul>        
    """
}

FUNDAMENTOS["Aranovich"] = {
    "full": """
    <h1 style="color:#00E5C0;">Aranovich</h1>
    <p style="text-align: justify; font-size:16px;">
     The Aranovich isotherm model is a theoretical extension of the BET model, developed to more accurately describe polymolecular adsorption on homogeneous surfaces, particularly in porous and dispersed materials. Like the BET model, it is based on the concept of multilayer adsorption; however, it introduces modifications that allow for a more realistic consideration of the interactions between adsorbed molecules and the variation in adsorption energy with the number of layers formed.

Unlike the BET model, which assumes constant energies for the upper layers, the Aranovich model incorporates thermodynamic corrections that account for additional free energy configurations, thereby improving the representation of the adsorption process over a wider range of concentrations. This allows for overcoming some limitations of the BET model, especially in systems where lateral interactions and saturation effects cannot be neglected.

Consequently, the Aranovich model is particularly useful for determining the specific surface area and for describing systems where multilayer adsorption deviates from ideal behavior.

    </p>
    <p style="text-align: center;">
    <img src="ARANOVICH.png" style="height: 60px;">
    </p>
    
       
    <p style="text-align: justify; font-size:16px;"> 
    where <i>q<sub>mA</sub></i> is the maximum adsorption capacity (mg/g), <i>C<sub>A</sub></i> is the Aranovich constant, <i>C<sub>SA</sub></i> is the saturation concentration (mg/L).
    </p>
         
    
    <p style="text-align: justify;color:#00E5C0; font-size:18px;"><b>Interpretation:</b></p>
    <ul>
    
    <p style="font-size:16px;">
The <b> <i>q<sub>mA</sub></i> (maximum adsorption capacity), </b> represents the maximum amount of adsorbate required to cover the surface of the adsorbent.
</p>

<ul style="font-size:16px;">
<li><b><i>q<sub>mA</sub></i> high</b> → higher adsorption capacity / larger surface area.</li>
<li><b><i>q<sub>mA</sub></i> low</b> → lower adsorption capacity / larger surface area</li>
</ul>

    
    <p style="font-size:16px;">
The <b> <i>C<sub>A</sub></i> (Aranovich constant), </b> a constant associated with the system’s affinity and interactions in multilayer adsorption.
</p>

<ul style="font-size:16px;">
<li><b><i>C<sub>A</sub></i> high</b> → Strong adsorbate–adsorbent interaction</li>
<li><b><i>C<sub>A</sub></i> low</b> → Weak adsorbate–adsorbent interaction</li>
</ul>
    """
}



# ==================================================
# DIALOGO PARA SELECCIONAR MODELOS 
# ==================================================
class SelectorModelos(QDialog):
    def __init__(self, modelos, seleccionados=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select models to display")
        self.setMinimumWidth(440)
        layout = QVBoxLayout()
        info = QLabel("<span style='color:black;'>Select the models you want to display in the graph:</span>")
        layout.addWidget(info)

        self.checkboxes = []
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        for nombre in modelos:
            cb = QCheckBox(nombre)
            if seleccionados is None:
                cb.setChecked(True)
            else:
                cb.setChecked(nombre in seleccionados)
            self.checkboxes.append(cb)
            scroll_layout.addWidget(cb)
        scroll_widget.setLayout(scroll_layout)
        scroll_widget.setStyleSheet("QCheckBox { color: #0D1B2A; font-size: 14px; }")
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        btn_ok = QPushButton("Show selected curves")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)

        self.setLayout(layout)

    def get_seleccionados(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class AdsorptionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IsoSorb — Isotherm Fitting (Nonlinear)                                        Universidad Autónoma del Estado de México - Tecnológico Nacional de México Campus Tapachula.")

        self._build_ui()
        self.show_home()
        self._apply_dark_theme()

        QtCore.QTimer.singleShot(0, self.adjust_window)

    def adjust_window(self):
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width()*0.9), int(screen.height()*0.9))

        qr = self.frameGeometry()
        cp = QtWidgets.QApplication.primaryScreen().geometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
        self.df = None
        self.mode = "Isotherms"  
        self._last_fit = None
        self._last_many_fits = None
        self.selected_overlay_models = list(MODELOS_ISOTERMAS.keys())

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # menú / controles
        left = QtWidgets.QFrame()
        left.setMaximumWidth(420)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignTop)
        
        # ===== LOGO DE ISOSORB =====
        self.logo = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(resource_path("IsoSorb.png"))

        self.logo.setPixmap(
            pixmap.scaled(
                260, 260,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )

        self.logo.setAlignment(QtCore.Qt.AlignCenter)

        left_layout.addWidget(self.logo)



        title = QtWidgets.QLabel("📊 Adsorption Adjustment")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#00E5C0;")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Selección de Isotermas
        self.mode_group = QtWidgets.QGroupBox("Select a language")
        self.mode_group.hide()
        mg_layout = QtWidgets.QHBoxLayout()
        self.rb_iso = QtWidgets.QRadioButton("Isotherms")
        self.rb_iso.setChecked(True)
        self.mode_group.setLayout(mg_layout)
        left_layout.addWidget(self.mode_group)
        
        


        # CARGAR  CSV
        self.load_btn = QtWidgets.QPushButton("📁 Upload data (CSV)")
        self.load_btn.clicked.connect(self.load_csv)
        left_layout.addWidget(self.load_btn)
    

        # infromación de como deben nombrarse las columas del CSV
        self.info_label = QtWidgets.QLabel("Isotherms: Ceq (mg/L), qeq (mg/g)\nKinetics: t (min), qt (mg/g)")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

       
        self.model_combo = QtWidgets.QComboBox()
        self.refresh_models()

       
        self.model_combo.setStyleSheet("""
        QComboBox {
        color: #00FF99;
        background-color: #1b1b1b;
        border: 1px solid #00FF99;
        padding: 5px;
        font-weight: bold;
        }
        QComboBox QAbstractItemView {
        background-color: #1b1b1b;
        color: #00FF99;
        selection-background-color: #00AA77;
        selection-color: white;
        }
        """)

        # -----------------------------------
        # FUNDAMENTOS
        # -----------------------------------
        model_layout = QtWidgets.QHBoxLayout()

        self.model_combo.setMinimumWidth(180)

        self.fund_btn = QtWidgets.QPushButton("📘 Fundamentals")
        self.fund_btn.setToolTip("View details of the selected model")
        self.fund_btn.setFixedWidth(150)
        self.fund_btn.clicked.connect(self.show_fundamentos)

        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.fund_btn)

        left_layout.addLayout(model_layout)

        # Boton de ajuste
        self.fit_btn = QtWidgets.QPushButton("🔬 Fit model (nonlinear)")
        self.fit_btn.clicked.connect(self.fit_model)
        left_layout.addWidget(self.fit_btn)

        # Checkbox: seleccionar modelos
        self.overlay_chk = QtWidgets.QCheckBox("Show all curves on a single graph")
        left_layout.addWidget(self.overlay_chk)

        
        self.select_models_btn = QtWidgets.QPushButton("🧾 Select models for the graph")
        self.select_models_btn.clicked.connect(self.open_selector_dialog)
        left_layout.addWidget(self.select_models_btn)

       
        self.init_button = QtWidgets.QPushButton("⚙️ Show/hide initial parameters")
        self.init_button.setCheckable(True)
        self.init_button.clicked.connect(self.toggle_initials)
        left_layout.addWidget(self.init_button)

       
        self.init_area = QtWidgets.QWidget()
        QtWidgets.QFormLayout(self.init_area)
        self.init_edits = {}  
        self.init_area.setVisible(False)
        left_layout.addWidget(self.init_area)

       
        left_layout.addWidget(QtWidgets.QLabel("Results:"))
        self.results = QtWidgets.QTextEdit()
        self.results.setReadOnly(True)
        self.results.setFixedHeight(260)
        left_layout.addWidget(self.results)

        # ------------------------------------------------
        # BOTÓN DISCUSIÓN CIENTÍFICA..
        # ------------------------------------------------

        self.discussion_btn = QtWidgets.QPushButton("🧠 Scientific discussion")
        self.discussion_btn.clicked.connect(self.generate_scientific_discussion)
        left_layout.addWidget(self.discussion_btn)
        
        
        
        # ------------------------------------------------
        # BOTÓN INFORMACIÓN DEL SOFTWARE
        # ------------------------------------------------

        self.about_btn = QtWidgets.QPushButton("🧾 About the software")
        self.about_btn.clicked.connect(self.show_about)
        left_layout.addWidget(self.about_btn)

        left_layout.addStretch()
        main_layout.addWidget(left, 0)

     
        right = QtWidgets.QFrame()
        right_layout = QtWidgets.QVBoxLayout(right)

    
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)

       
        self.html_view = QTextBrowser()
        self.html_view.setOpenExternalLinks(True)
        self.html_view.setOpenLinks(False)
        self.html_view.anchorClicked.connect(self.handle_html_links)
       
        self.html_view.setStyleSheet("""
            QTextBrowser {
                background-color: #0E1111;
                color: #DFFFEA;
                border-radius: 8px;
                padding: 12px;
            }
            h2 { color: #00E5C0; }
            h3 { color: #8EF6D8; }
            p, li { color: #CFEFE6; }
            code { background-color: #0B2B20; color: #A7FFD1; padding: 2px 4px; border-radius:4px; }
        """)
        self.html_view.setVisible(False)
        right_layout.addWidget(self.html_view)

       
        hbtn_layout = QHBoxLayout()
        self.download_conc_btn = QtWidgets.QPushButton()
        self.download_conc_btn.setVisible(False)
        hbtn_layout.addWidget(self.download_conc_btn)
        hbtn_layout.addStretch()
        right_layout.addLayout(hbtn_layout)

        # ===========================================================
        # BOTONES FINALES (Guardar, Análisis, Limpiar)
        # ===========================================================
        # botón guardar
        btns = QtWidgets.QHBoxLayout()
        self.save_plot_btn = QtWidgets.QPushButton("💾 Save plot")
        self.save_plot_btn.clicked.connect(self.save_plot)
        btns.addWidget(self.save_plot_btn)
        
        # botón graficar residuos
        self.residual_btn = QtWidgets.QPushButton("📉 Residual plot")
        self.residual_btn.clicked.connect(self.show_residuals)
        btns.addWidget(self.residual_btn)

        #  botón: Análisis
        self.analysis_btn = QtWidgets.QPushButton("📊 Analysis")
        self.analysis_btn.setVisible(False)  # Oculto inicialmente
        self.analysis_btn.clicked.connect(self.show_analysis)
        btns.addWidget(self.analysis_btn)

        self.clear_btn = QtWidgets.QPushButton("🧹 Clean")
        self.clear_btn.clicked.connect(self.clear_all)
        btns.addWidget(self.clear_btn)
        right_layout.addLayout(btns)

        
        self.overlay_chk.stateChanged.connect(self.toggle_analysis_button)

        main_layout.addWidget(right, 1)
        self.setLayout(main_layout)

    def _apply_dark_theme(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(18, 18, 18))
        pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
        pal.setColor(QPalette.Base, QColor(25, 25, 25))
        pal.setColor(QPalette.Text, QColor(230, 230, 230))
        pal.setColor(QPalette.Button, QColor(45, 45, 45))
        pal.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(pal)
        self.setStyleSheet("""
            QWidget { color:#E0E0E0; font-family: 'Segoe UI', 'Roboto'; }
            QPushButton { background-color:#00C9A7; color:#000; border-radius:8px; padding:8px 10px; font-weight:600; }
            QPushButton:checked { background-color:#00E5C0; }
            QGroupBox { border:1px solid #00C9A7; border-radius:8px; padding:8px; }
            QComboBox, QLineEdit { background-color:#1E1E1E; border:1px solid #00C9A7; color:#EEE; padding:6px; border-radius:6px; }
            QTextEdit { background-color:#111; border-radius:10px; color:#00E5C0; }
            QLabel { color:#CFEFE6; }
        """)

   
    def refresh_models(self):
        self.model_combo.clear()
        self.model_combo.addItems(list(MODELOS_ISOTERMAS.keys()))
        self.info_label.setText("Expected columns in CSV → Ceq (mg/L), qeq (mg/g)")

    
    def toggle_initials(self):
        show = self.init_button.isChecked()
        self.init_area.setVisible(show)
        if show:
            self.populate_initials()

    def populate_initials(self):
        layout = self.init_area.layout()
        while layout.count():
            w = layout.takeAt(0)
            if w.widget():
                w.widget().deleteLater()
        self.init_edits = {}
        model = self.model_combo.currentText()

     
      
        if model in MODELOS_ISOTERMAS:
            _, defaults, _, labels = MODELOS_ISOTERMAS[model]
            for i, d in enumerate(defaults):
                name = f"p{i}"
                le = QtWidgets.QLineEdit(str(d))
                layout.addRow(f"{name}:", le)
                self.init_edits[name] = le
            return

        

    # -------------------------
    # CSV 
    # -------------------------
    def load_csv(self):
        
        self.hide_fundamentos()

        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"The CSV file could not be read:\n{e}")
            return
        if self.rb_iso.isChecked():
            expected = ["Ceq", "qeq"]
            if not all(c in df.columns for c in expected):
                QMessageBox.warning(self, "Error", f"The CSV file must contain columns: {expected}")
                return
            df = df.dropna(subset=expected)
            self.df = df[expected].astype(float).reset_index(drop=True)
            self.results.append(f"Isothermal data loaded: {len(self.df)} points.")
            self.plot_data(self.df["Ceq"].values, self.df["qeq"].values, xlabel="Ceq (mg/L)", ylabel="qeq (mg/g)")
        
    
    def fit_model(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "<span style='color:black;'> First, upload a CSV file containing data.</span>")
            return

        if self.overlay_chk.isChecked():
            self.fit_and_plot_all_models()
            return

        model = self.model_combo.currentText()
        if self.rb_iso.isChecked():
            x = self.df["Ceq"].values
            y = self.df["qeq"].values
            xlabel, ylabel = "Ceq (mg/L)", "qeq (mg/g)"
       

        p0 = None
        if self.init_edits:
            try:
                p0 = [float(self.init_edits[k].text()) for k in sorted(self.init_edits.keys())]
            except:
                p0 = None

        popt = None; pcov = None; err = None

        if self.rb_iso.isChecked() and model in MODELOS_ISOTERMAS:
            func, defaults, bounds, labels = MODELOS_ISOTERMAS[model]
            if p0 is None:
                p0 = defaults
            popt, pcov, err = safe_curve_fit(func, x, y, p0=p0, bounds=bounds)
       
        else:
            QMessageBox.warning(self, "Error", "Unrecognized model.")
            return

        if err is not None:
            QMessageBox.warning(self, "Failed adjustment", f"It could not be adjusted: {err}")
            return

        try:
            fitted = func(x, *popt)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error evaluating the fitted model: {e}")
            return

        stats = compute_statistics(y, fitted, n_params=len(popt))

        self.results.clear()
        self.results.append(f"Model: {model}")
        if self.rb_iso.isChecked() and model in MODELOS_ISOTERMAS:
            labels = MODELOS_ISOTERMAS[model][3]
        
        else:
            labels = []
        for i, val in enumerate(popt):
            label = labels[i] if i < len(labels) else f"p[{i}]"
            self.results.append(f"  {label} = {val:.6g}")
        self.results.append("")
        self.results.append("Statistical parameters:")
        self.results.append(f"  R²     = {stats['R2']:.6f}")
        self.results.append(f"  SSE    = {stats['SSE']:.6f}")
        self.results.append(f"  SAE    = {stats['SAE']:.6f}")
        self.results.append(f"  ARE    = {stats['ARE']:.6f}")
        self.results.append(f"  X²     = {stats['X2']:.6f}")
        self.results.append(f"  RMSE   = {stats['RMSE']:.6f}")
        self.results.append(f"  HYBRID = {stats['HYBRID']:.6f}")
        self.results.append(f"  MPSD   = {stats['MPSD']:.6f}")

        self.plot_data_with_fit(x, y, fitted, model=model, popt=popt, xlabel=xlabel, ylabel=ylabel)

        self._last_fit = {
            "mode": ("Isotherms" if self.rb_iso.isChecked() else "Kinetics"),
            "model": model,
            "params": popt,
            "stats": stats
        }

    def _legend_text(self, model, popt):
        if model in MODELOS_ISOTERMAS:
            labels = MODELOS_ISOTERMAS[model][3]
       
        else:
            labels = []
        pairs = []
        for i, val in enumerate(popt):
            lab = labels[i] if i < len(labels) else f"p{i}"
            short = lab.split('[')[0].strip()
            pairs.append(f"{short}={val:.3g}")
        return f"{model}: " + ", ".join(pairs)

    # -------------------------
    # Graficar
    # -------------------------
    def clear_plot(self):
        self.fig.clear()
        self.canvas.draw()

    def plot_data(self, x, y, xlabel="", ylabel=""):
        self.hide_fundamentos()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.scatter(x, y, marker='o', edgecolors='k', label="Data")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        self.canvas.draw()

    def plot_data_with_fit(self, x, y, fitted_y, model=None, popt=None, xlabel="x", ylabel="y"):
        self.hide_fundamentos()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.scatter(x, y, marker='o', edgecolors='k', label="Data")
        sorted_idx = np.argsort(x)
        ax.plot(x[sorted_idx], fitted_y[sorted_idx], '-', linewidth=2, label=self._legend_text(model, popt))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        self.canvas.draw()

   
    def open_selector_dialog(self):
        if self.rb_iso.isChecked():
            modelos = list(MODELOS_ISOTERMAS.keys())
        
        dlg = SelectorModelos(modelos, seleccionados=self.selected_overlay_models, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            sel = dlg.get_seleccionados()
            self.selected_overlay_models = sel

   
    def fit_and_plot_all_models(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "<span style='color:black;'> First, upload a CSV file containing data.</span>")
            return

        if self.rb_iso.isChecked():
            x = self.df["Ceq"].values
            y = self.df["qeq"].values
            xlabel, ylabel = "Ceq (mg/L)", "qeq (mg/g)"
            available_models = list(MODELOS_ISOTERMAS.keys())
            model_map = MODELOS_ISOTERMAS
       

        modelos_a_probar = [m for m in self.selected_overlay_models if m in available_models]
        if not modelos_a_probar:
            modelos_a_probar = available_models

        results_list = []
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.scatter(x, y, marker='o', edgecolors='k', label="Data")
        xs = np.linspace(min(x), max(x), 400)

        for model in modelos_a_probar:
            func, defaults, bounds, labels = model_map[model]
            p0 = defaults
            try:
                popt, pcov, err = safe_curve_fit(func, x, y, p0=p0, bounds=bounds)
                if err:
                    continue
                y_pred_smooth = func(xs, *popt)
                y_pred_on_x = func(x, *popt)
            except Exception:
                continue

            stats = compute_statistics(y, y_pred_on_x, n_params=len(popt))
            results_list.append({
                "model": model,
                "params": popt,
                "stats": stats
            })
            ax.plot(xs, y_pred_smooth, linewidth=2, label=self._legend_text(model, popt))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        self.canvas.draw()

       
        self.results.clear()
        mode_name = 'Isotherms' if self.rb_iso.isChecked() else 'Kinetics'
        self.results.append(f"Results: Fitting of selected models for mode: {mode_name}")
        self.results.append("")
        for r in results_list:
            self.results.append(f"Model: {r['model']}")
            labels = MODELOS_ISOTERMAS[r['model']][3] if r['model'] in MODELOS_ISOTERMAS else None
            for i, val in enumerate(r['params']):
                lab = labels[i] if i < len(labels) else f"p[{i}]"
                self.results.append(f"  {lab} = {val:.6g}")
            s = r['stats']
            self.results.append(f"  R²     = {s['R2']:.6f}")
            self.results.append(f"  SSE    = {s['SSE']:.6f}")
            self.results.append(f"  SAE    = {s['SAE']:.6f}")
            self.results.append(f"  ARE    = {s['ARE']:.6f}")
            self.results.append(f"  X²     = {s['X2']:.6f}")
            self.results.append(f"  HYBRID = {s['HYBRID']:.6f}")
            self.results.append(f"  MPSD   = {s['MPSD']:.6f}")
            self.results.append(f"  RMSE   = {s['RMSE']:.6f}")
            self.results.append("")

       
        self._last_many_fits = results_list
    
  
        self._last_many_fits = compute_sne(self._last_many_fits)
  
    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save chart", "ajuste.png", "PNG Files (*.png);;All Files (*)")
        if not path:
            return
        self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor())
        QMessageBox.information(self, "Saved", f"Graph saved in:\n{path}")

    def clear_all(self):
        self.df = None
        self.results.clear()
        self.clear_plot()
        self._last_fit = None
        self._last_many_fits = None
        self.hide_fundamentos()

    # -------------------------
    # Export to Excel
    # -------------------------
    def export_to_excel(self):
        rows = []
        if getattr(self, "_last_many_fits", None):
            for r in self._last_many_fits:
                model = r["model"]
                params = r["params"]
                stats = r["stats"]
                labels = MODELOS_ISOTERMAS[model][3] if model in MODELOS_ISOTERMAS else None
                row = {"model": model}
                for i, val in enumerate(params):
                    row[f"param_{i}"] = val
                    row[f"param_{i}_label"] = labels[i] if i < len(labels) else f"p[{i}]"
                row.update(stats)
                rows.append(row)
        elif getattr(self, "_last_fit", None):
            r = self._last_fit
            model = r["model"]
            params = r["params"]
            stats = r["stats"]
            labels = MODELOS_ISOTERMAS[model][3] if model in MODELOS_ISOTERMAS else None
            row = {"model": model}
            for i, val in enumerate(params):
                row[f"param_{i}"] = val
                row[f"param_{i}_label"] = labels[i] if i < len(labels) else f"p[{i}]"
            row.update(stats)
            rows.append(row)
        else:
            QMessageBox.information(self, "Export", "There are no fitting results to export. Fit a model or check 'Show all curves''.")
            return

        df_out = pd.DataFrame(rows)
        path, _ = QFileDialog.getSaveFileName(self, "Save results to Excel", "resultados_ajuste.xlsx", "Excel Files (*.xlsx);;All Files (*)")
        if not path:
            return
        try:
            df_out.to_excel(path, index=False)
            QMessageBox.information(self, "Exported", f"Results exported to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Error exporting", f"The export failed:\n{e}")

    # -------------------------
    # ANÁLISIS 
    # -------------------------
    def toggle_analysis_button(self, state):
        self.analysis_btn.setVisible(state == Qt.Checked)

    def show_analysis(self):
        if not getattr(self, "_last_many_fits", None):
            QMessageBox.information(self, "Analysis", "<span style='color:black;'> First, run “Show All Curves” (setting) to generate analysis data.</span>")
            return

        ordenados = sorted(self._last_many_fits, key=lambda r: r['stats']['SNE'])

        dlg = QDialog(self)
        dlg.setWindowTitle("📊 Automatic Analysis: Top Models")
        dlg.setMinimumSize(1600, 900)
        dlg_layout = QtWidgets.QVBoxLayout(dlg)
        dlg.setStyleSheet("""
            QDialog { background-color: #121212; color: #E6FFFA; }
            QLabel { color: #A7FFE7; font-weight:600; }
            QTableWidget { background-color: #1B1B1B; color: #00FF99; gridline-color: #333; }
            QPushButton { background-color: #00C9A7; color:#000; border-radius:6px; padding:6px; }
        """)

        title = QLabel("Comparison of adsorption models")
        dlg_layout.addWidget(title)

        table = QTableWidget(len(ordenados), 10)        
        header = table.horizontalHeader()

        font = header.font()
        font.setBold(True)
        header.setFont(font)

        table.setStyleSheet("""
                        QHeaderView::section {
                            background-color: #EAEAEA;
                            color: black;
                            font-weight: bold;
                            }
                        """)
        
        
        
        
        table.setHorizontalHeaderLabels([
    "Model","R²","SSE","SAE","ARE","X²","HYBRID","MPSD","RMSE","SNE"
])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        for i, r in enumerate(ordenados):
            model = r['model']
            s = r['stats']
            table.setItem(i,0,QTableWidgetItem(str(model)))
            table.setItem(i,1,QTableWidgetItem(f"{s['R2']:.6f}"))
            table.setItem(i,2,QTableWidgetItem(f"{s['SSE']:.6f}"))
            table.setItem(i,3,QTableWidgetItem(f"{s['SAE']:.6f}"))
            table.setItem(i,4,QTableWidgetItem(f"{s['ARE']:.6f}"))
            table.setItem(i,5,QTableWidgetItem(f"{s['X2']:.6f}"))
            table.setItem(i,6,QTableWidgetItem(f"{s['HYBRID']:.6f}"))
            table.setItem(i,7,QTableWidgetItem(f"{s['MPSD']:.6f}"))
            table.setItem(i,8,QTableWidgetItem(f"{s['RMSE']:.6f}"))
            table.setItem(i,9,QTableWidgetItem(f"{s['SNE']:.6f}"))

        dlg_layout.addWidget(table)

        best = ordenados[0]
        best_model = best['model']
        best_SNE = best['stats']['SNE']
        best_R2 = best['stats']['R2']
        params = best.get("params", [])

        conclusion = ""
        conclusion += f"Best model according to SNE: {best_model}\n"
        conclusion += f"SNE = {best_SNE:.6f} | R² = {best_R2:.6f}\n\n"

        if best_R2 >= 0.99:
            conclusion += "Excellent fit: the model fits the experimental data very well.\n"
        elif best_R2 >= 0.97:
            conclusion += "Very good fit: the model accurately describes the system's behavior.\n"
        elif best_R2 >= 0.95:
            conclusion += "Good fit: The model is acceptable, although there may be minor deviations.\n"
        else:
            conclusion += "Moderate or poor fit: the model does not fully describe the experimental behavior.\n"

        # ---------------------------
        # Interpretación automática del modelo
        # ---------------------------

        interpretacion = ""

        if best_model.lower() == "langmuir":
            interpretacion += (
                "The Langmuir model suggests that adsorption occurs in a monolayer "
                "on a homogeneous surface with equivalent adsorption sites "
                "and without interaction between adsorbed molecules.\n"
            )

        elif best_model.lower() == "freundlich":
            interpretacion += (
                "The Freundlich model describes adsorption on heterogeneous surfaces "
                "with different adsorption energies.\n"
            )

        elif best_model.lower() == "sips":
            interpretacion += (
                "The Sips model combines the Langmuir and Freundlich models, describing "
                "heterogeneous systems that, at high concentrations, tend toward a monolayer.\n"
            )

        elif best_model.lower() == "toth":
            interpretacion += (
                "The Toth model is a modification of the Langmuir model for "
                "heterogeneous systems where deviations from ideal behavior occur.\n"
            )

        elif best_model.lower() == "temkin":
            interpretacion += (
                "The Temkin model posits that the heat of adsorption decreases "
                "linearly with increasing surface coverage.\n"
            )

        elif best_model.lower() == "dubinin-radushkevich":
            interpretacion += (
                "The Dubinin–Radushkevich model describes adsorption in "
                "microporous materials and distinguishes between physical and chemical adsorption.\n"
            )

        elif best_model.lower() == "bet":
            interpretacion += (
                "The BET model describes multilayer adsorption on solid surfaces.\n"
            )

        else:
            interpretacion += (
                "El The selected model adequately describes the adsorption system.\n"
            )
            
        

        conclusion += "\nInterpretation of the model:\n"
        conclusion += interpretacion
        
        # ------------------------------------------------
        # ANÁLISIS DEL MECANISMO DE ADSORCIÓN
        # ------------------------------------------------

        mecanismo = ""

        modelo = best_model.lower()
        modelo = modelo.replace("–", "-")
        modelo = modelo.replace("ó", "o")
        modelo = modelo.replace("_", "-")

        # ------------------------------------------------
        # LINEAL (LEY DE HENRY)
        # ------------------------------------------------
        if "lineal" in modelo or "linear" in modelo:

            if len(params) >= 1:

                KH = params[0]

                mecanismo += (
                    "The linear model corresponds to Henry's law and "
                    "describes proportional adsorption between the amount "
                    "adsorbed and the concentration of the adsorbate.\n"
                )

                mecanismo += (
                    f"The constant KH = {KH:.4f} represents the affinity "
                    "adsorption of the adsorbate by the adsorbent at low concentrations.\n"
                )

        # ------------------------------------------------
        # LANGMUIR
        # ------------------------------------------------
        elif "langmuir" in modelo:

            if len(params) >= 2:

                qmax = params[0]
                KL = params[1]

                mecanismo += (
                    "The Langmuir model describes monolayer adsorption "
                    "across a homogeneous surface with equivalent locations.\n"
                )

                mecanismo += (
                    f"The maximum capacity qmax = {qmax:.4f} represents "
                    "the saturation of the active sites of the adsorbent.\n"
                )

                mecanismo += (
                    f"The constant KL = {KL:.4f} indicates the affinity "
                    "between the adsorbate and the adsorbent.\n"
                )

        # ------------------------------------------------
        # FREUNDLICH
        # ------------------------------------------------
        elif "freundlich" in modelo:

            if len(params) >= 2:

             
                n = params[1]

                mecanismo += (
                    "The Freundlich model describes adsorption on "
                    "heterogeneous surfaces with different adsorption energies.\n"
                )

                if n > 1:

                    mecanismo += (
                        f"The parameter n = {n:.3f} indicates favorable adsorption "
                        "and a predominance of physical interactions.\n"
                    )

                else:

                    mecanismo += (
                        f"The parameter n = {n:.3f} indicates less favorable adsorption.\n"
                    )

        # ------------------------------------------------
        # TEMKIN
        # ------------------------------------------------
        elif "temkin" in modelo:

            if len(params) >= 2:

               
                B = params[1]

                mecanismo += (
                    "The Temkin model considers that the heat of adsorption "
                    "decreases linearly as coverage increases "
                    "superficial due to adsorbate–adsorbent interactions.\n"
                )

                mecanismo += (
                    f"The parameter B = {B:.4f} is related to "
                    "the adsorption energy of the system.\n"
                )

        # ------------------------------------------------
        # DUBININ-RADUSHKEVICH
        # ------------------------------------------------
        elif "radushkevich" in modelo:

            if len(params) >= 2:

                beta = params[1]

                mecanismo += (
                    "EThe Dubinin–Radushkevich model describes adsorption "
                    "in microporous materials.\n"
                )

                try:

                    E = (1 / (2 * abs(beta))) ** 0.5

                    if E < 8:

                        mecanismo += (
                            f"The average energy E = {E:.2f} kJ/mol indicates "
                            "physical adsorption.\n"
                        )

                    elif 8 <= E <= 16:

                        mecanismo += (
                            f"The energy E = {E:.2f} kJ/mol suggests "
                            "ion exchange.\n"
                        )

                    else:

                        mecanismo += (
                            f"The energy E = {E:.2f} kJ/mol indicates "
                            "chemical adsorption.\n"
                        )

                except:

                    mecanismo += (
                        "It was not possible to calculate the average adsorption energy.\n"
                    )

        # ------------------------------------------------
        # REDLICH-PETERSON
        # ------------------------------------------------
        elif "redlich" in modelo:

            if len(params) >= 3:

                beta = params[2]

                mecanismo += (
                    "The Redlich–Peterson model is a hybrid model "
                    "that combines features of the Langmuir and Freundlich models.\n"
                )

                if abs(beta - 1) < 0.2:

                    mecanismo += (
                        f"β = {beta:.3f} indicates behavior close to "
                        "the Langmuir model.\n"
                    )

                else:

                    mecanismo += (
                        f"β = {beta:.3f} indicates heterogeneity "
                        "on the adsorbent surface.\n"
                    )

        # ------------------------------------------------
        # SIPS
        # ------------------------------------------------
        elif "sips" in modelo:

            if len(params) >= 3:

                n = params[2]

                mecanismo += (
                    "The Sips model combines the Langmuir and Freundlich models to "
                    "describe heterogeneous surfaces.\n"
                )

                if n < 1:

                    mecanismo += (
                        f"The parameter n = {n:.3f} indicates "
                        "energy heterogeneity on the surface of the adsorbent.\n"
                    )

                else:

                    mecanismo += (
                        f"n ≈ {n:.3f} suggests behavior close to "
                        "the Langmuir model.\n"
                    )

        # ------------------------------------------------
        # TÓTH
        # ------------------------------------------------
        elif "toth" in modelo:

            if len(params) >= 3:

                t = params[2]

                mecanismo += (
                    "The Tóth model describes adsorption on "
                    "heterogeneous surfaces with deviations from the Langmuir model.\n"
                )

                if t < 1:

                    mecanismo += (
                        f"The parameter t = {t:.3f} indicates "
                        "energetic heterogeneity at the adsorption sites.\n"
                    )

                else:

                    mecanismo += (
                        f"t ≈ {t:.3f} suggests behavior close to "
                        "the Langmuir model.\n"
                    )

        # ------------------------------------------------
        # DUBININ-ASTAKHOV
        # ------------------------------------------------
        elif "astakhov" in modelo:

            if len(params) >= 3:

                n = params[2]

                mecanismo += (
                    "The Dubinin–Astakhov model describes adsorption in "
                    "microporous materials with an energy distribution.\n"
                )

                mecanismo += (
                    f"The parameter n = {n:.3f} describes the heterogeneity "
                    "of the adsorption sites.\n"
                )

        # ------------------------------------------------
        # VOLMER
        # ------------------------------------------------
        elif "volmer" in modelo:

            mecanismo += (
                "The Volmer model describes adsorption by considering "
                "interactions between molecules adsorbed on the surface.\n"
            )

        # ------------------------------------------------
        # BET
        # ------------------------------------------------
        elif "bet" in modelo:

            mecanismo += (
                "The BET model describes multilayer adsorption "
                "on solid surfaces.\n"
            )

        # ------------------------------------------------
        # ARANOVICH
        # ------------------------------------------------
        elif "aranovich" in modelo:

            mecanismo += (
                "The Aranovich model describes multilayer adsorption "
                "by taking into account interactions between adsorbed layers.\n"
            )

        # ------------------------------------------------
        # OTROS MODELOS
        # ------------------------------------------------
        else:

            mecanismo += (
                f"No automatic translation is available "
                f"for the {best_model} model. It is recommended to analyze "
                "the model parameters for interpreting the mechanism.\n"
            )

        conclusion += "\nProbable mechanism of adsorption:\n"
        conclusion += mecanismo
        
        conclusion += "\nRelevant statistical information:\n"
        conclusion += "- R² measures the correlation between experimental and predicted values.\n"
        conclusion += "- SSE represents the sum of the squares of the errors.\n"
        conclusion += "- SAE measures the total absolute error between experimental and calculated values.\n"
        conclusion += "- ARE represents the model's average relative error.\n"
        conclusion += "- χ² assesses the discrepancy between experimental and predicted values.\n"
        conclusion += "- HYBRID improves the sensitivity of the adjustment in low-concentration ranges.\n"
        conclusion += "- MPSD considers the model's degrees of freedom to evaluate the error.\n"
        conclusion += "- RMSE represents the average deviation between experimental and predicted data.\n"
        conclusion += "- SNE includes multiple error functions to select the best model.\n"
        
       # ------------------------------------------------
        # ANÁLISIS DE RESIDUOS
        # ------------------------------------------------

        try:

            Ce = self.df["Ceq"].values
            q_exp = self.df["qeq"].values

            ajuste = None

            for r in self._last_many_fits:

                if r["model"] == best_model:

                    ajuste = r
                    break


            if ajuste is None:

                conclusion += "\nThe model fit for calculating residuals was not found.\n"

            else:

                params = ajuste["params"]

                modelo_func = None

                if best_model in MODELOS_ISOTERMAS:

                    modelo_func = MODELOS_ISOTERMAS[best_model][0]

                
                if modelo_func is None:

                    conclusion += "\nThe model function could not be obtained.\n"

                else:

                    q_pred = modelo_func(Ce, *params)

                    residuos = q_exp - q_pred

                    sigma = np.std(residuos)

                    conclusion += "\nResidue analysis:\n"

                    if sigma == 0:

                        conclusion += "The standard deviation of the residuals could not be calculated.\n"

                    else:

                        outliers = np.where(np.abs(residuos) > 3 * sigma)[0]

                        if len(outliers) > 0:

                            

                            conclusion += (
                                f"{len(outliers)} possible outliers were detected in the experimental data.\n"
                                f"at the experimental sites: .\n"
                            )

                        else:

                            conclusion += (
                                "No significant outliers were detected in the data.\n"
                            )

        except Exception as e:

            conclusion += f"\nError calculating residuals: {e}\n"


        lbl_conc = QtWidgets.QLabel(conclusion)
        lbl_conc.setWordWrap(True)
        dlg_layout.addWidget(lbl_conc)

        btns = QtWidgets.QHBoxLayout()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        btns.addWidget(btn_close)

        def export_summary():

            rows = []

            for r in ordenados:
                row = {"model": r["model"]}
                stats = r["stats"]

                for k, v in stats.items():
                    row[k] = v

                rows.append(row)

            df_out = pd.DataFrame(rows)

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save analysis summary",
                "summary_analysis.xlsx",
                "Excel Files (*.xlsx);;All Files (*)"
            )

            if not path:
                return

            try:
                df_out.to_excel(path, index=False)
                QMessageBox.information(self, "Exported", f"Summary saved in:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"The export failed:\n{e}")

        btn_export = QPushButton("💾 Export summary")
        btn_export.clicked.connect(export_summary)
        btns.addWidget(btn_export)

        dlg_layout.addLayout(btns)

        dlg.exec_()


    # -------------------------
    # GRAFICA DE RESIDUOS
    # -------------------------

    def show_residuals(self):

        ajuste = None

        # Caso 1: se usó "mostrar todas las curvas"
        if getattr(self, "_last_many_fits", None):

            ordenados = sorted(self._last_many_fits, key=lambda r: r['stats']['SNE'])

            ajuste = ordenados[0]

        # Caso 2: se ajustó solo un modelo
        elif getattr(self, "_last_fit", None):

            ajuste = self._last_fit

        else:

            QMessageBox.information(
                self,
                "Residuals", "<span style='color:black;'>First, fit a model or use 'Show All Curves'.</span>"
            )
            return


        best_model = ajuste["model"]

        params = ajuste["params"]


        Ce = self.df["Ceq"].values
        q_exp = self.df["qeq"].values


        if best_model in MODELOS_ISOTERMAS:

            modelo_func = MODELOS_ISOTERMAS[best_model][0]

        
        else:

            QMessageBox.warning(self, "Error", "The model function could not be obtained")
            return


        q_pred = modelo_func(Ce, *params)

        residuos = q_exp - q_pred


        self.fig.clear()

        ax = self.fig.add_subplot(111)

        ax.scatter(Ce, residuos)

        ax.axhline(0)

        ax.set_xlabel("Equilibrium concentration (Ce)")

        ax.set_ylabel("residuals (qeq - qmodelo)")

        ax.set_title(f"Residual plot - {best_model}")


        # ---------------------------------------
        # INTERPRETACIÓN AUTOMÁTICA
        # ---------------------------------------

        sigma = np.std(residuos)

        interpretacion = ""

        if sigma < 0.05:

            interpretacion = "The model exhibits a low error and provides a good representation of the experimental data. It is recommended to verify that the residuals are randomly distributed around zero."

        elif sigma < 0.15:

            interpretacion = "The model exhibits a moderate error and is capable of representing the adsorption system. It is recommended to analyze the distribution of the residuals to confirm the validity of the fit."

        else:

            interpretacion = "The model presents a relatively high error, which suggests that it may not adequately describe the adsorption process or that experimental deviations may be influencing the results."


        outliers = np.where(np.abs(residuos) > 3 * sigma)[0]

        if len(outliers) > 0:

            interpretacion += f"\nA total of {len(outliers)} potential outliers were identified."


        texto = (
            f"Modelo: {best_model}\n"
            f"σ residuos = {sigma:.4f}\n\n"
            f"{interpretacion}"
        )


        ax.text(
            0.05,
            0.95,
            texto,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )


        self.canvas.draw()

    
    def generate_scientific_discussion(self):

        if not getattr(self, "_last_many_fits", None):

            QMessageBox.information(
                self,
                "Discussion", "<span style='color:black;'>First, run the model fitting.</span>"
            )
            return


        # ordenar por SNE (mejor modelo)
        ordenados = sorted(self._last_many_fits, key=lambda r: r['stats']['SNE'])

        best = ordenados[0]

        best_model = best["model"]
        best_stats = best["stats"]


        # ------------------------------------------------
        # TABLA COMPARATIVA DE MODELOS
        # ------------------------------------------------

        filas = ""

        for r in ordenados:

            m = r["model"]
            s = r["stats"]

            r2 = s.get("R2",0)
            rmse = s.get("RMSE",0)
            sse = s.get("SSE",0)
            sne = s.get("SNE",0)
            chi = s.get("CHI2",0)

            filas += f"""
            <tr>
            <td>{m}</td>
            <td>{r2:.4f}</td>
            <td>{rmse:.4f}</td>
            <td>{sse:.4f}</td>
            <td>{chi:.4f}</td>
            <td>{sne:.6f}</td>
            </tr>
            """


        # ------------------------------------------------
        # DETERMINAR TIPO DE ADSORCIÓN (D-R)
        # ------------------------------------------------

        tipo_ads = "Undetermined"

        for r in ordenados:

            if r["model"].lower() == "dubinin–radushkevich" or r["model"].lower() == "dubinin-radushkevich":

                B = r["params"][1]

                if B > 0:

                    E = (1 / (2*B))**0.5

                    if E < 8:
                        tipo_ads = "Physical adsorption"
                    elif 8 <= E <= 16:
                        tipo_ads = "Ion exchange"
                    else:
                        tipo_ads = "Chemical adsorption"

                break


        r2 = best_stats.get("R2",0)
        rmse = best_stats.get("RMSE",0)
        sne = best_stats.get("SNE",0)


        # ------------------------------------------------
        # TEXTO CIENTÍFICO
        # ------------------------------------------------

        html = f"""

        <div style="background-color:#0E1111;
                    color:#E8FFF7;
                    font-family:'Segoe UI';
                    padding:20px;">

        <h1 style="color:#00E5C0;">Discussion of the adsorption process</h1>

        <p>
        The experimental data were analyzed by means of nonlinear fitting using various adsorption isotherm models. The optimal model was selected based on multiple error functions, including R², RMSE, SSE, χ², and SNE.
        </p>

        <p>
        Among the evaluated models, <b>{best_model}</b> demonstrated the best statistical performance, presenting the lowest SNE value (<b>{sne:.6f}</b>) and a coefficient of determination of <b>R² = {r2:.4f}.
        </p>

        <p>
        The obtained RMSE value (<b>{rmse:.4f}</b>) indicates that the model predictions show relatively low deviations from the experimental data.
        </p>


        <h2 style="color:#00E5C0;">Statistical comparison of models</h2>

        <table border="1" cellpadding="6" cellspacing="0"
        style="border-collapse:collapse;color:white;">

        <tr style="background-color:#003c2f;">
        <th>Model</th>
        <th>R²</th>
        <th>RMSE</th>
        <th>SSE</th>
        <th>χ²</th>
        <th>SNE</th>
        </tr>

        {filas}

        </table>


        <h2 style="color:#00E5C0;">Type of adsorption</h2>

        <p>
        The analysis based on the Dubinin–Radushkevich model enables the estimation of the mean adsorption energy and, consequently, the identification of the dominant adsorption mechanism.
        </p>

        <p>
According to this analysis, the system exhibits
<b>{tipo_ads}</b>.
</p>


        <h2 style="color:#00E5C0;">Scientific conclusion</h2>

<p>
Overall, the results obtained indicate that the selected model adequately describes the behavior of the evaluated adsorption system. The comparison of error functions confirms that the selection of the optimal model should be based on the simultaneous analysis of multiple statistical criteria.
</p>

</div>
        """


        self.canvas.setVisible(False)

        self.html_view.setHtml(html)

        self.html_view.setVisible(True)


        self._discussion_html = html
    
    
    # ------------------------------------------------
    # PANTALLA INICIAL
    # ------------------------------------------------

    def show_home(self):

        html = """
        <div style="background-color:#0E1111;
                    color:#E8FFF7;
                    font-family:'Segoe UI';
                    padding:25px;">

        <h1 style="color:#00E5C0;">IsoSorb: Computational Tool for Adsorption Isotherm Modeling</h1>
        <p style="text-align: center;">
        <img src="IsoSorbX.png" >
        </p>

        <p style="font-size:18px;">
        Scientific software developed for the analysis of adsorption processes based on nonlinear fitting of isotherm models.
        </p>

        <h2 style="color:#00E5C0;">What does this software do?</h2>

        <ul>
    <p style="font-size:18px;">
        <li>Fitting of multiple adsorption models</li>
        <li>Advanced statistical comparison</li>
        <li>Residual analysis</li>
        <li>Outlier detection</li>
        <li>Automatic interpretation</li>
        <li>Automatic generation of scientific discussion (paper-style)</li>
    </p>
</ul>

<h2 style="color:#00E5C0;">Getting started</h2>

<p style="font-size:18px;">
    1. Load CSV file<br>
    2. Select model<br>
    3. Run fitting<br>
    4. Analyze results<br>
</p>

<br><br>

<a href="manual" style="
    background-color:#00E5C0;
    color:#000;
    padding:12px 20px;
    text-decoration:none;
    border-radius:8px;
    font-weight:bold;">
    📘 Download user manual (PDF)
</a>

</div>
        """

        self.canvas.setVisible(False)
        self.html_view.setHtml(html)
        self.html_view.setVisible(True)

    def handle_html_links(self, url):

        if url.toString() == "manual":
            self.download_manual()
    
    def download_manual(self):

        import os
        import shutil
        import subprocess

        origen = os.path.join(os.getcwd(), "manual.pdf")

        destino, _ = QFileDialog.getSaveFileName(
            self,
            "“Save manual”",
            "adsorption_manual.pdf",
            "PDF Files (*.pdf)"
        )

        if not destino:
            return

        try:
            shutil.copy(origen, destino)

            # 📂 
            if os.name == 'nt':  #
                os.startfile(destino)
            elif os.name == 'posix':
                subprocess.call(('xdg-open', destino))

            QMessageBox.information(
                self,
                "Full download",
                "The manual was saved and opened successfully."
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error",
                f"<span style='color:black;'> The manual could not be downloaded:</span>\n{e}"
            )
    
    # -------------------------
    # FUNDAMENTOS & PDF
    # -------------------------

    def show_fundamentos(self):
        """
        Muestra en el área de la derecha (QTextBrowser) los fundamentos en HTML
        del modelo seleccionado.
        """

        mode_iso = self.rb_iso.isChecked()
        model = self.model_combo.currentText()

        source = FUNDAMENTOS

        html = ""
        conclusion_text = ""

        if model in source:
            html = source[model]["full"]
            conclusion_text = source[model].get("conclusion", "")

        else:
            if mode_iso:

                keys = list(MODELOS_ISOTERMAS.keys())

                html_parts = [
                    "<h1 style='color:#00E5C0;'>Fundamentals — Isotherms</h1>"
                ]

                for k in keys:
                    if k in source:
                        html_parts.append(source[k]["full"])

                html = "\n".join(html_parts)
                conclusion_text = "Compilation of conclusions regarding isotherms."

            

                for k in keys:
                    if k in source:
                        html_parts.append(source[k]["full"])

                html = "\n".join(html_parts)
                conclusion_text = "Compilation of kinetic findings."

        if not html:
            html = "<h3>No basis was found for the selected model.</h3>"
            conclusion_text = ""

        full_html = f"""
        <div style="font-family: 'Segoe UI'; line-height:1.3;">
          {html}
          <hr style="border: 1px solid #073b2a;">
          <p>{conclusion_text}</p>
        </div>
        """

        self.canvas.setVisible(False)
        self.html_view.setHtml(full_html)
        self.html_view.setVisible(True)
        self.download_conc_btn.setVisible(True)

        if conclusion_text:
            self._current_conclusion_html = f"<h2>{model}</h2><p>{conclusion_text}</p>"
        else:
            self._current_conclusion_html = ""


    def hide_fundamentos(self):

        self.html_view.clear()
        self.html_view.setVisible(False)
        self.canvas.setVisible(True)
        self.download_conc_btn.setVisible(False)
        self._current_conclusion_html = ""


    def export_conclusions_pdf(self):
        """
        Exporta las conclusiones mostradas a PDF
        """

        if not getattr(self, "_current_conclusion_html", None):

            QMessageBox.information(
                self,
                "Export",
                "<span style='color=black;'>There are no conclusions to draw.</span>"
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save conclusions (PDF)",
            "conclusions_model.pdf",
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not path:
            return

        doc = QTextDocument()
        doc.setDefaultFont(self.font())

        doc.setHtml(f"""
        <html>
        <head>
        <meta charset="utf-8"/>
        <style>
        body {{font-family:'Segoe UI'; color:#002b20;}}
        h2 {{color:#004d33;}}
        p {{color:#0b3b2d;}}
        </style>
        </head>

        <body>
        {self._current_conclusion_html}
        </body>

        </html>
        """)
        
        

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageMargins(12, 12, 12, 12, QPrinter.Millimeter)

        try:

            doc.print_(printer)

            QMessageBox.information(
                self,
                "Exported",
                f"Conclusions exported to:\n{path}"
            )

        except Exception as e:

            QMessageBox.warning(
                self,
                "Error exporting",
                f"The file could not be exported to PDF:\n{e}"
            )

    # ------------------------------------------------
    # INFORMACIÓN DEL SOFTWARE (EN PANEL PRINCIPAL)
    # ------------------------------------------------

    def show_about(self):

        html = """

        <div style="
    background-color:#0E1111;
    color:#E8FFF7;
    font-family:'Segoe UI';
    padding:20px;
    line-height:1.4;
    ">

    <h1 style="color:#00E5C0; font-size:26px;">
        Adsorption Analysis Tool
    </h1>

    <h2 style="color:#00E5C0; font-size:20px;">Version</h2>

    <p style="font-size:18px;">
        <b>IsoSorb 1.0</b> - Software developed for the analysis of adsorption processes through
        nonlinear fitting of isotherm models, incorporating
        advanced statistical evaluation, residual analysis, and automatic
        outlier detection.
    </p>


<h2 style="color:#00E5C0; font-size:20px; text-align:left;">
Authors
</h2>

<h2 style="color:#00E5C0; font-size:20px; text-align:left;">
Authors
</h2>

<div style="font-size:18px; text-align:left;">

<ul>
    <li>Dagoberto Arce Córdova<sup>a,b</sup></li>
    <li>Edith Erielia  Gutiérrez Segura<sup>b</sup></li>
    <li>Ruben fernando Gutiérrez Hernández<sup>c</sup></li>
    <li>J. E. Moreno Marcelino<sup>b</sup></li>
    <li>Javier Francisco Valle Mora<sup>c</sup></li>
    <li>Jose Luis Maldonado<sup>c</sup></li>
    <li>Diego Albero Ulloa Gutierrez<sup>c</sup></li>
    <li>Nery del carmen Rodiles Cruz<sup>c</sup></li>
    <li>Rosember Hidalgo López<sup>c</sup></li>
</ul>

<p>
<sup>a</sup> Universidad Autónoma del Estado de México. Facultad de Química. Doctoral Program in Materials Science, Paseo Colón esq. Paseo Tollocan s/n, 50120, Toluca, México, CP 50120, Mexico.
</p>

<p>
<sup>b</sup> Facultad de Química, Universidad Autónoma del Estado de México, Paseo Colón esq. Paseo Tollocan s/n, Toluca, Mexico, C.P. 50120, Mexico.
</p>

<p>
<sup>c</sup> Instituto Tecnológico de Tapachula / Tecnológico Nacional de Mexico, Department of Chemical and Biochemical Engineering, Carretera a Puerto Madero Km 2, Tapachula 30700, Chiapas, Mexico.
</p>

<p>
E-mail: rub.gutierrez@tapachula.tecnm.mx
</p>

</div>

</div>



</p>


    <h2 style="color:#00E5C0; font-size:20px;">Repository</h2>

    <p style="font-size:18px;">
        <a style="color:#7CFFD8;" href="https://github.com/dagoarcecord/IsoSorb">
            https://github.com/dagoarcecord/IsoSorb
          </a>
        
            
    </p>


    <h2 style="color:#00E5C0; font-size:20px;">License</h2>

    <p style="font-size:18px;">
        GNU General Public License v3.0 
    </p>


    <h2 style="color:#00E5C0; font-size:20px;">Included Isotherm Models</h2>

    <ul style="font-size:18px;">
        <li>Linear</li>
        <li>Langmuir</li>
        <li>Freundlich</li>
        <li>Temkin</li>
        <li>Dubinin–Radushkevich</li>
        <li>Redlich–Peterson</li>
        <li>Sips</li>
        <li>Tóth</li>
        <li>Dubinin–Astakhov</li>
        <li>Volmer</li>
        <li>BET</li>
        <li>Aranovich</li>
    </ul>


    <h2 style="color:#00E5C0; font-size:20px;">Statistical Functions</h2>

    <ul style="font-size:18px;">
        <li>R²</li>
        <li>SSE</li>
        <li>SAE</li>
        <li>ARE</li>
        <li>χ²</li>
        <li>HYBRID</li>
        <li>MPSD</li>
        <li>RMSE</li>
        <li>SNE</li>
    </ul>


    <h2 style="color:#00E5C0; font-size:20px;">Software Features</h2>

    <ul style="font-size:18px;">
        <li>Nonlinear model fitting</li>
        <li>Automatic comparison of multiple models</li>
        <li>Automatic selection of the best model</li>
        <li>Adsorption mechanism analysis</li>
        <li>Fitting plots</li>
        <li>Residual analysis</li>
        <li>Automatic outlier detection</li>
        <li>Export of results to Excel</li>
        <li>Export of conclusions to PDF</li>
    </ul>


    <h2 style="color:#00E5C0; font-size:20px;">How to cite the software (APA)</h2>

    <p style="font-size:18px;">
        Arce-Córdova, D., Gutiérrez Segura, E., Gutiérrez Hernández, R. F., Moreno Marcelino, J. E., Valle Mora, J. F., Maldonado, J. L., Ulloa Gutierrez, D. A., Rodiles-Cruz, N. C., & Hidalgo-López, R. (2026). IsoSorb: A Python-based tool for nonlinear adsorption isotherm modeling and multi-criteria model selection (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20150901
    </p>

</div>

        """

        self.canvas.setVisible(False)

        self.html_view.setHtml(html)
        self.html_view.setVisible(True)

        self.download_conc_btn.setVisible(False)
# ==================================================
# MAIN
# ==================================================

def main():
    import sys
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    win = AdsorptionApp()

    screen = QtWidgets.QApplication.primaryScreen().availableGeometry()

    w = 1100
    h = 800

    x = (screen.width() - w) // 2
    y = (screen.height() - h) // 2

    win.setGeometry(x, y, w, h)

    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
