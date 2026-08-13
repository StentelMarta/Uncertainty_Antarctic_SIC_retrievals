import pandas as pd
import numpy as np
import csv
import os
import sys

import smrt
from smrt import make_atmosphere
from smrt import PSU
from smrt.permittivity.ice import ice_permittivity_maetzler06
from smrt.permittivity.generic_mixing_formula import polder_van_santen
from smrt.inputs.make_medium import make_snow_layer, make_ice_column, make_ice_layer, make_water_body, make_transparent_volume
from smrt import make_snowpack, sensor_list, PSU, make_model, make_interface

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, 'my_smrt_models'))
from my_smrt_models.seawater_permittivity_meissner_wentz import seawater_permittivity_meissner_wentz


# ---------------------------------------------------------------------------
# Atmosphere definition
# ---------------------------------------------------------------------------

file_path19 = os.path.join(BASE_DIR, 'Ocean-reference-model-mirror', 'Outputs', 'TbAtmo_19_0deg.dat')
file_path37 = os.path.join(BASE_DIR, 'Ocean-reference-model-mirror', 'Outputs', 'TbAtmo_37_0deg.dat')
file_path89 = os.path.join(BASE_DIR, 'Ocean-reference-model-mirror', 'Outputs', 'TbAtmo_89_0deg.dat')


def load_tb_data(filepath, max_theta=75):
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        na_values='*****',
        comment='$',
        engine='python'
    )
    df.columns = df.columns.str.strip().str.rstrip(',')
    df = df[df['theta(deg)'] <= max_theta]
    return {
        'theta': df['theta(deg)'].to_list(),
        'tb_down': df['TbDown(K)'].to_list(),
        'tb_up': df['TbUp(k)'].to_list(),
        'transmittance': np.exp(-df['tau(neper)']).to_list()
    }


data37 = load_tb_data(file_path37)
data19 = load_tb_data(file_path19)
data89 = load_tb_data(file_path89)

atmos19 = make_atmosphere("simple_atmosphere",
                          theta=data19['theta'],
                          tb_down=data19['tb_down'],
                          tb_up=data19['tb_up'],
                          transmittance=data19['transmittance'])

atmos37 = make_atmosphere("simple_atmosphere",
                          theta=data37['theta'],
                          tb_down=data37['tb_down'],
                          tb_up=data37['tb_up'],
                          transmittance=data37['transmittance'])


# ---------------------------------------------------------------------------
# Snowpack data import and SMRT configuration
# ---------------------------------------------------------------------------

base_output_path = os.path.join(BASE_DIR, 'OUTPUT_SMRT')

table_location = os.path.join(BASE_DIR, 'Reference_table_summer_references.csv')
sheet_data = pd.read_csv(table_location)

frequencies = {
    "37": 36.5,
    "19": 18.7
}


def extract_layer_parameters(sheet_data, layer_keywords):
    extracted_parameters = {}
    for layer_prefix in layer_keywords:
        filtered_data = sheet_data[
            sheet_data['Parameter'].str.contains(f'^[A-Za-z_]*_{layer_prefix}$', regex=True, na=False)
        ]
        extracted_parameters[layer_prefix] = {
            row['Parameter']: {'Value': row['Value REF']} for _, row in filtered_data.iterrows()
        }
    return extracted_parameters


def extract_parameters_with_min_max(sheet_data, layer_keywords, parameter_to_test):
    extracted_parameters = {}
    for layer_prefix in layer_keywords:
        filtered_data = sheet_data[
            sheet_data['Parameter'].str.contains(f'^[A-Za-z_]*_{layer_prefix}$', regex=True, na=False)
        ]
        layer_data = {}
        for _, row in filtered_data.iterrows():
            param_name = row['Parameter']
            if param_name == parameter_to_test:
                layer_data[param_name] = {
                    'Min': float(row['Min']),
                    'Max': float(row['Max']),
                    'Value': float(row['Value REF'])
                }
            else:
                layer_data[param_name] = {'Value': float(row['Value REF'])}
        extracted_parameters[layer_prefix] = layer_data
    return extracted_parameters


parameters_to_test = [
    "Thickness_SP", "Density_SP", "Temperature_SP", "Radius_SP",
    "Salinity_SP", "Fraction_volume_water_SP",
    "Thickness_DH", "Density_DH", "Temperature_DH", "Radius_DH",
    "Salinity_DH", "Fraction_volume_water_DH",
    "Thickness_SI", "Density_SI", "Temperature_SI", "Radius_SI",
    "Salinity_SI", "Fraction_volume_water_SI",
    "Thickness_SI_1", "Temperature_SI_1", "Salinity_SI_1",
]

layer_keywords = ["SP", "DH", "SI", "SI_1", "SI_2"]

N = 21


def create_snowpack_and_run_model(
    layer_parameters, frequency, parameter_to_test,
    atmos19, atmos37,
    N=21, output_file="results.csv"
):
    min_val = max_val = None

    for layer_key in ["SP", "DH", "SI", "SI_1", "SI_2"]:
        param = layer_parameters.get(layer_key, {}).get(parameter_to_test)
        if isinstance(param, dict) and "Min" in param:
            min_val = param["Min"]
            max_val = param["Max"]
            break

    if min_val is None or max_val is None or min_val == max_val:
        raise ValueError(
            f"Invalid Min/Max for '{parameter_to_test}': Min={min_val}, Max={max_val}."
        )

    test_values = np.linspace(float(min_val), float(max_val), N)
    results = []

    for value in test_values:
        for layer_key in ["SP", "DH", "SI", "SI_1", "SI_2"]:
            if parameter_to_test in layer_parameters.get(layer_key, {}):
                layer_parameters[layer_key][parameter_to_test]["Value"] = float(value)
                break

        sp_params = {k: v["Value"] for k, v in layer_parameters["SP"].items()}
        dh_params = {k: v["Value"] for k, v in layer_parameters["DH"].items()}
        si_params = {k: v["Value"] for k, v in layer_parameters["SI"].items()}
        si1_params = {k: v["Value"] for k, v in layer_parameters["SI_1"].items()}
        si2_params = {k: v["Value"] for k, v in layer_parameters["SI_2"].items()}

        density_ice = 917

        ssa_sp = 3 / (sp_params["Radius_SP"] * density_ice)
        porod_length_sp = 4 * (1 - sp_params["Density_SP"] / density_ice) / (ssa_sp * density_ice)

        ssa_dh = 3 / (dh_params["Radius_DH"] * density_ice)
        porod_length_dh = 4 * (1 - dh_params["Density_DH"] / density_ice) / (ssa_dh * density_ice)

        ssa_si = 3 / (si_params["Radius_SI"] * density_ice)
        porod_length_si = 4 * (1 - si_params["Density_SI"] / density_ice) / (ssa_si * density_ice)

        eps_saline_water_sp = seawater_permittivity_meissner_wentz(
            frequency * 1e9,
            271.25,
            sp_params["Salinity_SP"] * PSU
        )
        eps_ice_sp = ice_permittivity_maetzler06(frequency * 1e9, 271.25)
        eps_sp = polder_van_santen(sp_params["Fraction_volume_water_SP"], eps_ice_sp, eps_saline_water_sp)

        eps_saline_water_dh = seawater_permittivity_meissner_wentz(
            frequency * 1e9,
            271.25,
            dh_params["Salinity_DH"] * PSU
        )
        eps_ice_dh = ice_permittivity_maetzler06(frequency * 1e9, dh_params["Temperature_DH"])
        eps_dh = polder_van_santen(dh_params["Fraction_volume_water_DH"], eps_ice_dh, eps_saline_water_dh)

        eps_saline_water_si = seawater_permittivity_meissner_wentz(
            frequency * 1e9,
            271.25,
            si_params["Salinity_SI"] * PSU
        )
        eps_ice_si = ice_permittivity_maetzler06(frequency * 1e9, si_params["Temperature_SI"])
        eps_si = polder_van_santen(si_params["Fraction_volume_water_SI"], eps_ice_si, eps_saline_water_si)

        sp_top = make_snowpack(
            thickness=[sp_params["Thickness_SP"]],
            density=[sp_params["Density_SP"]],
            temperature=[sp_params["Temperature_SP"]],
            microstructure_model="unified_scaled_exponential",
            porod_length=porod_length_sp,
            polydispersity=0.7,
            radius=[sp_params["Radius_SP"]],
            ice_permittivity_model=eps_sp,
        )

        dh_layer = make_snowpack(
            thickness=[dh_params["Thickness_DH"]],
            density=[dh_params["Density_DH"]],
            temperature=[dh_params["Temperature_DH"]],
            microstructure_model="unified_scaled_exponential",
            porod_length=porod_length_dh,
            polydispersity=1.5,
            radius=[dh_params["Radius_DH"]],
            ice_permittivity_model=eps_dh,
        )

        si_layer = make_snowpack(
            thickness=[si_params["Thickness_SI"]],
            density=[si_params["Density_SI"]],
            temperature=[si_params["Temperature_SI"]],
            microstructure_model="unified_scaled_exponential",
            porod_length=porod_length_si,
            polydispersity=1.9,
            radius=[si_params["Radius_SI"]],
            brine_volume_fraction=0.05,
            ice_permittivity_model=eps_si,
        )

        ic1 = make_ice_column(
            "firstyear",
            thickness=[si1_params["Thickness_SI_1"]],
            microstructure_model="sticky_hard_spheres",
            stickiness=100,
            density=[si1_params["Density_SI_1"]],
            temperature=[si1_params["Temperature_SI_1"]],
            salinity=[si1_params["Salinity_SI_1"] * PSU],
            radius=[si1_params["Radius_SI_1"]],
            add_water_substrate=False,
        )

        ic2 = make_ice_column(
            "firstyear",
            thickness=[si2_params["Thickness_SI_2"]],
            microstructure_model="sticky_hard_spheres",
            stickiness=100,
            density=[si2_params["Density_SI_2"]],
            temperature=[si2_params["Temperature_SI_2"]],
            salinity=[si2_params["Salinity_SI_2"] * PSU],
            radius=[si2_params["Radius_SI_2"]],
            add_water_substrate=True,
        )

        total_snowpack19 = atmos19 + sp_top + dh_layer + si_layer + ic1 + ic2
        total_snowpack37 = atmos37 + sp_top + dh_layer + si_layer + ic1 + ic2

        sensor = sensor_list.amsr2("37" if frequency == 36.5 else "19")
        snowpack_model = make_model(
            "symsce_torquato21",
            "dort",
            rtsolver_options=dict(n_max_stream=128)
        )
        result_seaice = snowpack_model.run(
            sensor,
            total_snowpack37 if frequency == 36.5 else total_snowpack19
        )

        results.append({
            "Parameter": parameter_to_test,
            "Value": value,
            "TbH": result_seaice.TbH(),
            "TbV": result_seaice.TbV(),
        })

    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Parameter", "Value", "TbH", "TbV"])
        writer.writeheader()
        writer.writerows(results)

    return results
