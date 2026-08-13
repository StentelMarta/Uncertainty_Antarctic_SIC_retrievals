import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared coherent-layer SMRT model (grease ice / slush ice)
# ---------------------------------------------------------------------------

m_sce_coh = make_model(
    "symsce_torquato21",
    "dort",
    rtsolver_options=dict(
        n_max_stream=128,
        process_coherent_layers=True,
    ),
)


# ---------------------------------------------------------------------------
# Dark nilas — ice column structure
# ---------------------------------------------------------------------------

n_layers = 3
density_profile = [920, 920, 920]
radius_profile = [0.000002, 0.000002, 0.000002]
T_BVF_FIX = 269.0

SAL_RANGE = np.arange(5, 46, 1)
temperatures_sal = [269.0, 270.0, 271.0]
BVF_FIX = 0.2

results_bulk = []
for sal in SAL_RANGE:
    salinity_profile = [sal * PSU] * n_layers
    bvf = [BVF_FIX] * n_layers
    ic = make_ice_column(
        'firstyear',
        thickness=[0.02, 0.02, 0.01],
        microstructure_model='sticky_hard_spheres',
        stickiness=100,
        density=density_profile,
        temperature=temperatures_sal,
        salinity=salinity_profile,
        radius=radius_profile,
        add_water_substrate=True,
    )
    for i, layer in enumerate(ic.layers):
        layer.frac_volume = bvf[i]
        layer.brine_volume_fraction = bvf[i]
        layer.microstructure.frac_volume = bvf[i]
    res19 = snowpack_model.run(sensor19, atmos19 + ic)
    res37 = snowpack_model.run(sensor37, atmos37 + ic)
    results_bulk.append({'salinity': sal, 'TbV19': res19.TbV(), 'TbV37': res37.TbV()})

df_bulk = pd.DataFrame(results_bulk)

BVF_RANGE = np.arange(0.05, 0.20 + 0.005, 0.005)
S_FIX_BVF = 17.5
temperatures_bvf = [269.0, 270.0, 271.0]
salinity_profile_bvf = [S_FIX_BVF * PSU] * n_layers

results_bvf = []
for bvf_val in BVF_RANGE:
    ic = make_ice_column(
        'firstyear',
        thickness=[0.02, 0.02, 0.01],
        microstructure_model='sticky_hard_spheres',
        stickiness=100,
        density=density_profile,
        temperature=temperatures_bvf,
        salinity=salinity_profile_bvf,
        radius=radius_profile,
        add_water_substrate=True,
    )
    for i, layer in enumerate(ic.layers):
        layer.frac_volume = bvf_val
        layer.brine_volume_fraction = bvf_val
        layer.microstructure.frac_volume = bvf_val
    res19 = snowpack_model.run(sensor19, atmos19 + ic)
    res37 = snowpack_model.run(sensor37, atmos37 + ic)
    results_bvf.append({'bvf': bvf_val, 'TbV19': res19.TbV(), 'TbV37': res37.TbV()})

df_bvf = pd.DataFrame(results_bvf)

T_TOP_RANGE = np.arange(260.0, 271.25, 1.0)
T_BOTTOM = 271.25
S_FIX_T = 17
salinity_profile_T = [S_FIX_T * PSU] * n_layers
bvf_T = [brine_volume_cox83_lepparanta88(T_BVF_FIX, salinity_profile_T[i])
         for i in range(n_layers)]

results_T = []
for T_top in T_TOP_RANGE:
    temperatures_T = np.linspace(T_top, T_BOTTOM, n_layers).tolist()
    ic = make_ice_column(
        'firstyear',
        thickness=[0.02, 0.02, 0.01],
        microstructure_model='sticky_hard_spheres',
        stickiness=100,
        density=density_profile,
        temperature=temperatures_T,
        salinity=salinity_profile_T,
        radius=radius_profile,
        add_water_substrate=True,
    )
    for i, layer in enumerate(ic.layers):
        layer.frac_volume = bvf_T[i]
        layer.brine_volume_fraction = bvf_T[i]
        layer.microstructure.frac_volume = bvf_T[i]
    res19 = snowpack_model.run(sensor19, atmos19 + ic)
    res37 = snowpack_model.run(sensor37, atmos37 + ic)
    results_T.append({'T_top': T_top, 'TbV19': res19.TbV(), 'TbV37': res37.TbV()})

df_T = pd.DataFrame(results_T)


# ---------------------------------------------------------------------------
# Grease ice — slush-layer structure
# ---------------------------------------------------------------------------

water_layer = make_water_body(
    water_permittivity_model=seawater_permittivity_meissner_wentz,
    salinity=32 * PSU,
    temperature=273.15 - 1.9,
)

FIXED_THICKNESS_GREASE = 0.008
fractions_water = np.arange(0.60, 0.85, 0.05)

TbV19_lwf, TbV37_lwf = [], []
for lwc in fractions_water:
    ic = make_slush(
        thickness=FIXED_THICKNESS_GREASE,
        microstructure_model="sticky_hard_spheres",
        frac_liquid_water=lwc,
        ice_permittivity_model=ice_permittivity_maetzler06,
        water_permittivity_model=seawater_permittivity_meissner_wentz,
        radius=0.1e-3,
        salinity=32 * PSU,
        temperature=273.15 - 1.9,
        inclusion_shape='random_needles',
    ) + water_layer
    res19 = m_sce_coh.run(sensor19, atmos19 + ic)
    res37 = m_sce_coh.run(sensor37, atmos37 + ic)
    TbV19_lwf.append(float(res19.TbV()))
    TbV37_lwf.append(float(res37.TbV()))

TbV19_lwf = np.array(TbV19_lwf)
TbV37_lwf = np.array(TbV37_lwf)


# ---------------------------------------------------------------------------
# Slush ice — slush-layer structure
# ---------------------------------------------------------------------------

FIXED_LWF = 0.20
thicknesses_cm = np.array([1.0, 2.0, 3.0, 5.0,
                            8.0, 10.0, 15.0, 20.0, 25.0, 30.0])
thicknesses_m = thicknesses_cm / 100.0

TbV19_thk, TbV37_thk = [], []
for th in thicknesses_m:
    ic = make_slush(
        thickness=th,
        microstructure_model="sticky_hard_spheres",
        frac_liquid_water=FIXED_LWF,
        ice_permittivity_model=ice_permittivity_maetzler06,
        water_permittivity_model=seawater_permittivity_meissner_wentz,
        radius=0.1e-3,
        salinity=32 * PSU,
        temperature=273.15 - 1.9,
        inclusion_shape=None,
    ) + water_layer
    res19 = m_sce_coh.run(sensor19, atmos19 + ic)
    res37 = m_sce_coh.run(sensor37, atmos37 + ic)
    TbV19_thk.append(float(res19.TbV()))
    TbV37_thk.append(float(res37.TbV()))

TbV19_thk = np.array(TbV19_thk)
TbV37_thk = np.array(TbV37_thk)

FIXED_THICKNESS_M = 0.10
lwc_values = np.arange(0.15, 0.85, 0.05)

TbV19_lwc, TbV37_lwc = [], []
for lwc in lwc_values:
    ic = make_slush(
        thickness=FIXED_THICKNESS_M,
        microstructure_model="sticky_hard_spheres",
        frac_liquid_water=lwc,
        ice_permittivity_model=ice_permittivity_maetzler06,
        water_permittivity_model=seawater_permittivity_meissner_wentz,
        radius=0.1e-3,
        salinity=32 * PSU,
        temperature=273.15 - 1.9,
        inclusion_shape=None,
    ) + water_layer
    res19 = m_sce_coh.run(sensor19, atmos19 + ic)
    res37 = m_sce_coh.run(sensor37, atmos37 + ic)
    TbV19_lwc.append(float(res19.TbV()))
    TbV37_lwc.append(float(res37.TbV()))

TbV19_lwc = np.array(TbV19_lwc)
TbV37_lwc = np.array(TbV37_lwc)
