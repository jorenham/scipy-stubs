from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "N_A",
    "Avogadro",
    "Boltzmann",
    "Btu",
    "Btu_IT",
    "Btu_th",
    "G",
    "Julian_year",
    "Planck",
    "R",
    "Rydberg",
    "Stefan_Boltzmann",
    "Wien",
    "acre",
    "alpha",
    "angstrom",
    "arcmin",
    "arcminute",
    "arcsec",
    "arcsecond",
    "astronomical_unit",
    "atm",
    "atmosphere",
    "atomic_mass",
    "atto",
    "au",
    "bar",
    "barrel",
    "bbl",
    "blob",
    "c",
    "calorie",
    "calorie_IT",
    "calorie_th",
    "carat",
    "centi",
    "convert_temperature",
    "day",
    "deci",
    "degree",
    "degree_Fahrenheit",
    "deka",
    "dyn",
    "dyne",
    "e",
    "eV",
    "electron_mass",
    "electron_volt",
    "elementary_charge",
    "epsilon_0",
    "erg",
    "exa",
    "exbi",
    "femto",
    "fermi",
    "fine_structure",
    "fluid_ounce",
    "fluid_ounce_US",
    "fluid_ounce_imp",
    "foot",
    "g",
    "gallon",
    "gallon_US",
    "gallon_imp",
    "gas_constant",
    "gibi",
    "giga",
    "golden",
    "golden_ratio",
    "grain",
    "gram",
    "gravitational_constant",
    "h",
    "hbar",
    "hectare",
    "hecto",
    "horsepower",
    "hour",
    "hp",
    "inch",
    "k",
    "kgf",
    "kibi",
    "kilo",
    "kilogram_force",
    "kmh",
    "knot",
    "lambda2nu",
    "lb",
    "lbf",
    "light_year",
    "liter",
    "litre",
    "long_ton",
    "m_e",
    "m_n",
    "m_p",
    "m_u",
    "mach",
    "mebi",
    "mega",
    "metric_ton",
    "micro",
    "micron",
    "mil",
    "mile",
    "milli",
    "minute",
    "mmHg",
    "mph",
    "mu_0",
    "nano",
    "nautical_mile",
    "neutron_mass",
    "nu2lambda",
    "ounce",
    "oz",
    "parsec",
    "pebi",
    "peta",
    "pi",
    "pico",
    "point",
    "pound",
    "pound_force",
    "proton_mass",
    "psi",
    "pt",
    "quecto",
    "quetta",
    "ronna",
    "ronto",
    "short_ton",
    "sigma",
    "slinch",
    "slug",
    "speed_of_light",
    "speed_of_sound",
    "stone",
    "survey_foot",
    "survey_mile",
    "tebi",
    "tera",
    "ton_TNT",
    "torr",
    "troy_ounce",
    "troy_pound",
    "u",
    "week",
    "yard",
    "year",
    "yobi",
    "yocto",
    "yotta",
    "zebi",
    "zepto",
    "zero_Celsius",
    "zetta",
]

# mathematical constants
pi: float
golden: float
golden_ratio: float

# SI prefixes
quetta: float
ronna: float
yotta: float
zetta: float
exa: float
peta: float
tera: float
giga: float
mega: float
kilo: float
hecto: float
deka: float
deci: float
centi: float
milli: float
micro: float
nano: float
pico: float
femto: float
atto: float
zepto: float
yocto: float
ronto: float
quecto: float

# binary prefixes
# ruff: noqa: PYI054
kibi: Literal[0x400]
mebi: Literal[0x100000]
gibi: Literal[0x40000000]
tebi: Literal[0x10000000000]
pebi: Literal[0x4000000000000]
exbi: Literal[0x1000000000000000]
zebi: Literal[0x400000000000000000]
yobi: Literal[0x100000000000000000000]

# physical constants
c: float
speed_of_light: float
mu_0: float
epsilon_0: float
h: float
Planck: float
hbar: float
G: float
gravitational_constant: float
g: float
e: float
elementary_charge: float
R: float
gas_constant: float
alpha: float
fine_structure: float
N_A: float
Avogadro: float
k: float
Boltzmann: float
sigma: float
Stefan_Boltzmann: float
Wien: float
Rydberg: float

# mass in kg
gram: float
metric_ton: float
grain: float
lb: float
pound: float
blob: float
slinch: float
slug: float
oz: float
ounce: float
stone: float
long_ton: float
short_ton: float

troy_ounce: float
troy_pound: float
carat: float

m_e: float
electron_mass: float
m_p: float
proton_mass: float
m_n: float
neutron_mass: float
m_u: float
u: float
atomic_mass: float

# angle in rad
degree: float
arcmin: float
arcminute: float
arcsec: float
arcsecond: float

# time in second
minute: float
hour: float
day: float
week: float
year: float
Julian_year: float

# length in meter
inch: float
foot: float
yard: float
mile: float
mil: float
pt: float
point: float
survey_foot: float
survey_mile: float
nautical_mile: float
fermi: float
angstrom: float
micron: float
au: float
astronomical_unit: float
light_year: float
parsec: float

# pressure in pascal
atm: float
atmosphere: float
bar: float
torr: float
mmHg: float
psi: float

# area in meter**2
hectare: float
acre: float

# volume in meter**3
litre: float
liter: float
gallon: float
gallon_US: float
fluid_ounce: float
fluid_ounce_US: float
bbl: float
barrel: float

gallon_imp: float
fluid_ounce_imp: float

# speed in meter per second
kmh: float
mph: float
mach: float
speed_of_sound: float
knot: float

# temperature in kelvin
zero_Celsius: float
degree_Fahrenheit: float

# energy in joule
eV = elementary_charge
electron_volt = elementary_charge
calorie: float
calorie_th: float
calorie_IT: float
erg: float
Btu_th: float
Btu: float
Btu_IT: float
ton_TNT: float

# power in watt
hp: float
horsepower: float

# force in newton
dyn: float
dyne: float
lbf: float
pound_force: float
kgf = g
kilogram_force = g

def convert_temperature(val: npt.ArrayLike, old_scale: str, new_scale: str) -> np.float64 | npt.NDArray[np.float64]: ...
def lambda2nu(lambda_: npt.ArrayLike) -> np.float64 | npt.NDArray[np.float64]: ...
def nu2lambda(nu: npt.ArrayLike) -> np.float64 | npt.NDArray[np.float64]: ...
