# Finite Element Simulation and Biomaterial Selection Using COMSOL Multiphysics

Finite element analysis (FEA) of a **BCI-controlled powered lower-limb prosthesis** using **COMSOL Multiphysics 6.1**. The project covers structural mechanics, steady-state heat transfer, biomaterial selection and a thermal feature-extraction pipeline for data-driven analysis.

> **Context:** This work forms Chapter 8 of a graduation project on a Brain–Computer-Interface (BCI) controlled powered prosthesis, developed at Egypt-Japan University of Science and Technology (E-JUST).

---

## Table of Contents

- [Objective](#objective)
- [COMSOL Environment](#comsol-environment)
- [Governing Physics](#governing-physics)
- [Biomaterial Selection](#biomaterial-selection)
- [Simulation Campaign](#simulation-campaign)
  - [Ankle Foot with Hinge (Principal Case Study)](#ankle-foot-with-hinge-principal-case-study)
  - [Supporting Components](#supporting-components)
- [Thermal Feature Extraction](#thermal-feature-extraction)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [How to Reproduce](#how-to-reproduce)
- [Limitations and Future Work](#limitations-and-future-work)
- [Authors](#authors)
- [References](#references)

---

## Objective

Before committing the prosthesis components to manufacturing, each load-bearing part had to satisfy two requirements:

1. **Structural safety** — carry the forces generated during stance and actuation without yielding.
2. **Thermal safety** — remain within a safe temperature range so that neither the printed polymer nor the underlying human tissue is damaged by heat from the motor, lead screw and bearing assemblies.

---

## COMSOL Environment

| Module | Role |
|--------|------|
| COMSOL Multiphysics (core) | Finite-element engine, mesh generation, solvers and post-processing |
| Structural Mechanics — Solid Mechanics | Stress, strain and displacement under stance and actuator loads |
| Heat Transfer — Heat Transfer in Solids | Temperature distribution from motor, lead-screw and hinge heat sources |
| CAD Import Module | Direct import of SolidWorks part files |

### Modelling Workflow

1. **Global definitions** — loads, material constants and boundary values declared as named parameters
2. **Geometry** — SolidWorks CAD import
3. **Materials** — PETG, PLA or alloy steel bound to geometry domains
4. **Physics interfaces** — Solid Mechanics and Heat Transfer in Solids with boundary conditions
5. **Mesh** — free tetrahedral elements at chosen resolution
6. **Study** — stationary solver with optional parametric sweeps
7. **Results** — surface/volume plots, isosurfaces and derived values

---

## Governing Physics

### Structural Mechanics

Static equilibrium for a linear-elastic isotropic solid:

$$\nabla \cdot \boldsymbol{\sigma} + \mathbf{F}_V = 0$$

with Hooke's law relating stress and strain. The **von Mises equivalent stress** is compared against yield strength through:

$$\text{FOS} = \frac{\sigma_y}{\sigma_{vM}}$$

### Heat Transfer

Steady-state heat conduction:

$$\nabla \cdot (-k\,\nabla T) = Q$$

with convective boundary conditions:

$$q = h\,(T_{\text{ext}} - T)$$

### Link to Tissue Safety

- **Mechanical:** Interface pressures in prosthetic sockets range 12.5–760 kPa; sustained pressure above tissue tolerance causes ulcers and necrosis.
- **Thermal:** Prolonged skin contact above ~43–44 °C causes cumulative thermal injury (ISO 13732-1).

---

## Biomaterial Selection

### Classification of Prosthetic Biomaterials

| Family | Representative Materials | Typical Prosthetic Role |
|--------|--------------------------|------------------------|
| Metals | Titanium alloys, alloy steel | Structural joints, lead screw, load-bearing pins |
| Ceramics & Desiccants | Silica gel | Moisture control, electronics protection |
| Polymers | Silicone, TPU, PETG, PLA | Liners, sockets, printed structural shells |
| Composites | Carbon-fibre-reinforced polymers | Lightweight high-stiffness frames |

### Why PETG?

**PETG** (glycol-modified polyethylene terephthalate) was selected for the printed parts based on:

- Toughness and impact resistance superior to PLA
- Glass transition ~80 °C (vs. ~55–60 °C for PLA) — critical near warm actuator components
- Food-contact-grade, low-odour, skin-safe thermoplastic
- Tensile strength ~45–55 MPa, elastic modulus ~2 GPa

| Property | Symbol | Value | Unit |
|----------|--------|-------|------|
| Young's modulus | E | 2.1 × 10⁹ | Pa |
| Poisson ratio | ν | 0.38 | — |
| Yield strength | σ_y | 50 × 10⁶ | Pa |
| Density | ρ | 1270 | kg/m³ |
| Thermal conductivity | k | 0.20 | W/(m·K) |
| Heat capacity | C_p | 1200 | J/(kg·K) |

### Candidate Print Polymer Comparison

| Material | E (GPa) | σ_y (MPa) | T_g / HDT (°C) | k (W/m·K) | Note |
|----------|---------|-----------|-----------------|-----------|------|
| **PETG** | 2.0–2.2 | ~50 | 80 / 70 | 0.20 | **Selected** — tough, skin-safe, good T_g |
| PLA | 3.0–3.5 | ~55 | 55–60 / 50 | 0.13 | Stiff but low T_g near actuators |
| ABS | 1.8–2.5 | ~40 | 105 | 0.17 | Higher T_g; warps, fumes |
| Nylon (PA12) | 1.5–2.0 | ~45 | high | 0.25 | Tough; hygroscopic |
| TPU | 0.01–0.05 | ~25–40 | soft | 0.20 | Flexible; liners only |
| CFRP (PA-CF) | 5–8 | 60–100 | high | ~0.3 | Stiff/light; cost, abrasion |

---

## Simulation Campaign

### Ankle Foot with Hinge (Principal Case Study)

The central component of the ankle assembly — a PETG 3D-printed part connecting the powered crank and lead-screw actuator to the foot plate.

#### Geometry and Mesh

- Imported from SolidWorks via CAD Import Module
- 1 domain, 245 boundaries, 619 edges, 377 vertices
- Meshed volume: 8.19 × 10⁻⁴ m³
- Free tetrahedral mesh (Finer preset): **202,070 elements**, 42,742 vertices
- Quadratic (serendipity) displacement elements

#### Loads and Boundary Conditions

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Stance load | F_stance | 100 N (baseline) | Body load through the ankle |
| Max actuator force | F_actuator | 900 N | Peak lead-screw force |
| Ambient temperature | T_amb | 293.15 K (20 °C) | Surrounding air |
| Convection coefficient | h_air | 15 W/(m²·K) | Natural convection |
| Ground convection | h_ground | 50 W/(m²·K) | Sole-to-ground |
| Hinge heat flux | q_hinge | 30 W/m² | Bearing friction |
| Crank heat flux | q_crank | 50 W/m² (baseline) | Lead-screw contact heat |

#### Actuator: TD-8120MG Digital Servo

- 56 g carbon-brush DC servo, 4.8–7.2 V
- Stall torque: 20.5–22.8 kg·cm; stall current: 2.1–2.7 A
- Heat dissipation: 1.9 W (no-load) to 19.4 W (stall) at 7.2 V

#### Studies Performed

1. **Structural** — Solid Mechanics, sweeping F_stance over 100, 300, 600, 900 N
2. **Thermal (flux sweep)** — Heat Transfer, sweeping q_crank over 25, 50, 75, 100 W/m²
3. **Thermal (convection sweep)** — sweeping h_air over 5, 10, 15, 20 W/(m²·K)

#### Mesh Convergence

A refinement study from Normal to Extra-fine (45,958 → 550,783 elements) confirmed:
- **Displacement:** fully converged (< 1% variation)
- **Peak von Mises stress:** diverges at a sharp re-entrant corner — identified as a **stress singularity** (not a physical stress)

### Supporting Components

| Component | Material | Physics | Peak von Mises | FOS |
|-----------|----------|---------|----------------|-----|
| Foot with hinge (100 N) | PETG | Structural + Thermal | 3.20 MPa | 15.6 |
| Foot with hinge (900 N) | PETG | Structural + Thermal | 30.5 MPa | 1.6 |
| Powered hinge | PETG / PLA | Structural | Both elastic | > 1 |
| Lead-screw holder | Steel / PETG | Coupled struct. + thermal | Both safe | > 1 |
| Motor joint shaft | AISI 4340 | Structural | 9.63 MPa | > 40 |
| Electronics enclosure | PETG | Structural | 1.01 MPa | 49.6 |

---

## Thermal Feature Extraction

An automated Python pipeline converts COMSOL thermal exports into a structured dataset for data-driven analysis:

- **Input:** Temperature field (T, x, y, z) and heat-flux components per mesh node, plus scalar metadata (material, heat-source power, ambient temperature)
- **Output:** ~40 descriptive features per simulation, grouped into:
  - Temperature statistics (min, max, mean, std, skewness, kurtosis, percentiles)
  - Spatial gradient features
  - Heat-flux statistics
  - Domain-level metadata

---

## Key Results

### Structural

| F_stance (N) | Peak von Mises (MPa) | Factor of Safety | Max Displacement (mm) |
|:---:|:---:|:---:|:---:|
| 100 | 3.20 | 15.6 | 0.034 |
| 300 | 10.04 | 5.0 | 0.107 |
| 600 | 20.28 | 2.5 | 0.216 |
| 900 | 30.53 | 1.6 | 0.325 |

- Both stress and displacement scale linearly with load
- The part remains elastic across the entire range
- A stress singularity at the sharp rib-to-body corner produces mesh-dependent peak values — this is a numerical artefact, not a physical failure

### Thermal

| q_crank (W/m²) | T_max (°C) | T_mean (K) | Mean Flux (W/m²) |
|:---:|:---:|:---:|:---:|
| 25 | 33.1 | 293.45 | 4.22 |
| 50 | 46.3 | 293.65 | 6.83 |
| 75 | 59.4 | 293.86 | 9.43 |
| 100 | 72.6 | 294.07 | 12.03 |

- Peak temperature rises linearly with crank heat flux (~13 K per 25 W/m²)
- Hot region is confined to the crank/lead-screw end; foot body stays near ambient
- The hot spot is **conduction-limited** — increasing air convection has negligible effect
- At baseline (50 W/m²), T_max ≈ 46 °C — below the PETG heat-deflection temperature (~70 °C) and safe for tissue
- At 100 W/m², T_max ≈ 73 °C — reaches the PETG softening range, identifying the crank interface as the thermal bottleneck
- The PETG part can absorb < 0.3 W before softening; the servo must be thermally decoupled

---

## Repository Structure

```
├── README.md
├── models/
│   └── README.md                     # Model file details (mph file too large for GitHub)
├── plots/
│   ├── structural/                   # Von Mises stress fields and displacement plots
│   ├── thermal/                      # Temperature fields and heat flux plots
│   ├── mesh/                         # Mesh visualisation and convergence plots
│   ├── boundary_conditions/          # BC setup illustrations
│   └── parametric_sweeps/            # Stress vs. load, T_max vs. q_crank, T_max vs. h_air
├── docs/
│   └── servo_datasheet/              # TD-8120MG servo reference images
└── reports/
    └── Chapter_8_FINAL.pdf           # Full FEA chapter (PDF)
```

---

## How to Reproduce

### Requirements

- **COMSOL Multiphysics 6.1** (or later) with:
  - Structural Mechanics Module
  - Heat Transfer Module
  - CAD Import Module
- **Python 3.8+** with `pandas`, `numpy`, `scipy` (for thermal feature extraction)

### Steps

1. Open the `.mph` model file in COMSOL Multiphysics
2. Review the global parameters in the **Parameters** node
3. Run the **Stationary** study (structural) and the two **Parametric Sweep** studies (thermal)
4. Results are generated automatically in the **Results** node
5. Export temperature fields as spreadsheet data for the Python feature-extraction pipeline

---

## Limitations and Future Work

1. **Print anisotropy** — simulations use bulk isotropic properties; FDM printing produces anisotropic parts with inter-layer strength ~50–75% of in-plane strength
2. **Static loading only** — gait is cyclic (~10⁶ cycles/year); fatigue analysis was not performed
3. **Estimated boundary conditions** — heat inputs are prescribed surface fluxes, not measured motor power
4. **Mesh-dependent singularity** — peak stress at the sharp rib-to-body corner requires a design fillet
5. **No experimental validation yet** — proposed future work includes:
   - Dial-gauge / DIC measurement of tip deflection under calibrated load
   - Thermocouple readings at rib-root and crank-holder during powered operation

---

## Authors

- **Noran Morad**
- **Mariam Ihab**
- **Moustafa Abdullah**

---

## References

1. COMSOL Multiphysics Reference Manual, version 6.1, COMSOL AB, Stockholm, Sweden, 2022.
2. D. F. Williams, "On the mechanisms of biocompatibility," *Biomaterials*, vol. 29, no. 20, pp. 2941–2953, 2008.
3. F. P. Incropera, D. P. DeWitt, T. L. Bergman, and A. S. Lavine, *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley, 2011.
4. M. Zhang, A. R. Turner-Smith, A. Tanner, and V. C. Roberts, "Clinical investigation of the pressure and shear stress on the trans-tibial stump with a prosthesis," *Med. Eng. Phys.*, vol. 20, no. 3, pp. 188–198, 1998.
5. J. E. Sanders and C. H. Daly, "Normal and shear stresses on a residual limb in a prosthetic socket during ambulation," *J. Rehabil. Res. Dev.*, vol. 30, no. 2, pp. 210–221, 1993.
6. A. R. Moritz and F. C. Henriques, "Studies of thermal injury: II. The relative importance of time and surface temperature in the causation of cutaneous burns," *Am. J. Pathol.*, vol. 23, no. 5, pp. 695–720, 1947.
7. ISO 13732-1:2006, *Ergonomics of the Thermal Environment — Methods for the Assessment of Human Responses to Contact with Surfaces — Part 1: Hot Surfaces*, 2006.
8. ISO 10993-1:2018, *Biological Evaluation of Medical Devices — Part 1: Evaluation and Testing Within a Risk Management Process*, 2018.
9. B. D. Ratner, A. S. Hoffman, F. J. Schoen, and J. E. Lemons, *Biomaterials Science*, 3rd ed., Academic Press, 2013.
10. M. Geetha, A. K. Singh, R. Asokamani, and A. K. Gogia, "Ti based biomaterials — A review," *Prog. Mater. Sci.*, vol. 54, no. 3, pp. 397–425, 2009.
11. J. Barrios-Muriel et al., "Advances in orthotic and prosthetic manufacturing: A technology review," *Materials*, vol. 13, no. 2, art. 295, 2020.
12. K. Durgashyam et al., "Experimental investigation on mechanical properties of PETG material processed by FDM," *Mater. Today Proc.*, vol. 18, pp. 2052–2059, 2019.
13. M.-H. Hsueh et al., "Effect of printing parameters on the thermal and mechanical properties of 3D-printed PLA and PETG," *Polymers*, vol. 13, no. 11, art. 1758, 2021.
14. S. Farah, D. G. Anderson, and R. Langer, "Physical and mechanical properties of PLA — A comprehensive review," *Adv. Drug Deliv. Rev.*, vol. 107, pp. 367–392, 2016.
15. S. H. Ahn et al., "Anisotropic material properties of fused deposition modeling ABS," *Rapid Prototyp. J.*, vol. 8, no. 4, pp. 248–257, 2002.

---

## License

This project is part of a graduation project at [E-JUST](https://www.ejust.edu.eg/). All rights reserved.
