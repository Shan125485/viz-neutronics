########### inputs ################

## examples stored here

#### Gadolinium rods

# wtPercent235 = 0.012
# wtPercent235 = 0.019
wtPercent235 = 0.022
wtPercentGd = 0.08


# UO2
uo2_density = 10.96 * 0.95 # g/cm^3  https://www.nuclear-power.com/nuclear-power-plant/nuclear-fuel/fuel-assembly/fuel-pellets/uranium-dioxide-uo2/
U_235 = {"Mr": 235, "wtPerc": wtPercent235 }
U_238 = {"Mr": 238, "wtPerc": 1-wtPercent235}

U = { "Mr": None, "atPerc":1, "U-235": U_235, "U-238": U_238}
O_in_UO2 = { "Mr": 16, "atPerc": 2}
UO2 = { "wtPerc": 1-wtPercentGd,"density": uo2_density, "Mr": None, "U": U, "O in UO2": O_in_UO2}
# UO2 = { "wtPerc": 1, "density": uo2_density, "Mr": None, "U": U, "O in UO2": O_in_UO2}

# Gd2O3
Gd2O3_density = 7.41 # g/cm^3 https://www.azom.com/article.aspx?ArticleID=2353
Gd_152 =  {"Mr": 152, "wtPerc": 0.002 }
Gd_154 = {"Mr": 154, "wtPerc": 0.0218}
Gd_155 = {"Mr": 155, "wtPerc": 0.148}
Gd_156 =  {"Mr": 156, "wtPerc": 0.2047}
Gd_157 = {"Mr": 157, "wtPerc": 0.1565}
Gd_158 = {"Mr": 158, "wtPerc": 0.2484}
Gd_160 = {"Mr": 160, "wtPerc": 0.2186}

Gd = { "Mr": None, "atPerc": 2,  "Gd-152": Gd_152, "Gd-154": Gd_154, "Gd-155": Gd_155, "Gd-156": Gd_156, "Gd-157": Gd_157, "Gd-158": Gd_158, "Gd-160": Gd_160 }
O_in_Gd2O3 = { "Mr": 16, "atPerc": 3}


Gd2O3 = {"density": Gd2O3_density, "wtPerc": wtPercentGd , "Mr": None, "Gd": Gd, "O in Gd2O3": O_in_Gd2O3}

# Input
# input = { "Gd2O3": Gd2O3, "UO2": UO2  } 

##############################################################
# ENRICHED URANIUM INPUTS
# enrichment = 0.021
# enrichment = 0.032
enrichment = 0.024
U_235 = {"Mr": 235, "wtPerc": enrichment }
U_238 = {"Mr": 238, "wtPerc": 1-enrichment}
U_enriched = { "Mr": None, "atPerc":1, "U-235": U_235, "U-238": U_238}

UO2_enriched = { "wtPerc": 1,"density": uo2_density, "Mr": None, "U": U_enriched, "O in UO2": O_in_UO2}

# input = {"UO2": UO2_enriched  } 

### helium
He = {"Mr": 2, "atPerc": 1, "wtPerc": 1, "density": 6.33e-5}

# input = {'He': He}

## zirconium

Zr_90 = {"Mr": 90, "wtPerc": 0.5145 }
Zr_91 = {"Mr": 91, "wtPerc": 0.1122 }
Zr_92 = {"Mr": 92, "wtPerc": 0.1715 }
Zr_94 = {"Mr": 94, "wtPerc": 0.1738 }
Zr_96 = {"Mr": 96, "wtPerc": 0.028 }

Zr = {'Zr-90': Zr_90, 'Zr-91': Zr_91, 'Zr-92': Zr_92, 'Zr-94': Zr_94, 'Zr-96': Zr_96, 'density':6.52,  'wtPerc': 1, 'atPerc': 1}

# input = {'Zr': Zr,}

### M5 cladding #### https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_488.pdf
wtPercNb = 0.01
wtPercO = 0.0013
wtPercZr = 1-wtPercNb-wtPercO

Zr = {'Zr-90': Zr_90, 'Zr-91': Zr_91, 'Zr-92': Zr_92, 'Zr-94': Zr_94, 'Zr-96': Zr_96, 'wtPerc': wtPercZr  }
Nb = {'Mr': 93,'wtPerc': wtPercNb}
O = {'Mr': 16, 'wtPerc': wtPercO}
# input = {'density': 6.52, 'Zr': Zr, 'Nb': Nb, 'O': O}


##### Zircalloy-4 for guide thimbles #### https://www.atimaterials.com/Products/Documents/datasheets/zirconium/alloy/Zr_nuke_waste_disposal_v1.pdf, https://www.azom.com/article.aspx?ArticleID=7644
wtPercSn = 0.0145 # tin
wtPercFe = 0.002
wtPercCr = 0.001 # chromium
wtPercZr = 1-wtPercSn-wtPercFe - wtPercCr
Zr = {'Zr-90': Zr_90, 'Zr-91': Zr_91, 'Zr-92': Zr_92, 'Zr-94': Zr_94, 'Zr-96': Zr_96, 'wtPerc': wtPercZr  }
Sn = {'Mr': 119,'wtPerc': wtPercSn}
Fe = {'Mr': 56, 'wtPerc': wtPercFe}
Cr = {'Mr': 52,'wtPerc': wtPercCr}

# input = {'density': 6.56, 'Zr': Zr, 'Sn': Sn, 'Fe': Fe, 'Cr': Cr}



#### water at 300C
water_temp = 312.7 # Celsius

# Calculation for water density based on fixed reference of 0.7 g/cm^3 aat 300C
water_temp = water_temp + 273 # Kelvin
water_temp_ref = 312.7 # Celsius
water_temp_ref = water_temp_ref + 273 # Kelvin
water_density_ref = 0.69811515

waterDensity = water_temp_ref * water_density_ref / water_temp

O = { "Mr": 16, "atPerc": 1}
H = {"Mr": 1, "atPerc": 2}

# ppm 
PPM = 1175
B_10 = {"Mr": 10, "atPerc": 20}
B_11 = {"Mr": 11, "atPerc": 80}
B = {"B-10": B_10, "B-11": B_11, 'wtPerc': 1175 / 1e6}
H2O = {'O': O, 'H': H,  'wtPerc': 1}

H2O_boronated = {'H2O': H2O, 'B': B, 'density': waterDensity}
#input = H2O_boronated

## steel for RPV ### ASME A508 Grade 3   https://cdn.standards.iteh.ai/samples/28504/748d0804e90c4bc9ace68144efbd9b03/ASTM-A508-A508M-03.pdf
steelDensity = 7.75 # generic stainless steel

wtPercC = 0.0025
wtPercMn = 0.013
wtPercSi = 0.0027
wtPercNi = 0.007
wtPercCr = 0.0025
wtPercMo = 0.0052
balance = 1-wtPercC-wtPercMn-wtPercSi - wtPercNi - wtPercCr - wtPercMo

carbon = {'Mr': 12,'wtPerc': wtPercC}
manganese = {'Mr': 55,'wtPerc': wtPercMn}
silicon = {'Mr': 28,'wtPerc': wtPercSi}
nickel = {'Mr': 59,'wtPerc': wtPercNi}
chromium = {'Mr': 52,'wtPerc': wtPercCr}
molybdenum = {'Mr': 96,'wtPerc': wtPercMo}
Fe = {'Mr': 52,  'wtPerc': balance}
# input = {'Fe': Fe,'C': carbon, 'Mn': manganese, 'Si': silicon, 'Ni': nickel, 'Cr': chromium, 'Mo': molybdenum, 'density': steelDensity}


########### 1D reflector inputs ####################

### FUEL material
# The fuel is 1/3 UO2, 2/3 water by volume

volWater_volUO2 = 2 # ratio of water to UO2 volume, calculated from a single pin cell

volWater = volWater_volUO2 / (1 + volWater_volUO2)  # In the pin cell, what fraction of the total area does water cover? This will correspond to the effective density of water.
volUO2 = 1- volWater #   # In the pin cell, what fraction of the total area does UO2 cover? This will correspond to the effective density of UO2.


UO2_in_refl_model = {  "density": uo2_density, "U": U, "O in UO2": O_in_UO2,'wtPerc': volUO2 }
H2O_in_refl_model = { 'O': O, 'H': H,  'density': waterDensity , 'wtPerc': volWater}

# input = {'UO2': UO2_in_refl_model}
# input = {'H2O': H2O_in_refl_model }

### REFLECTOR material

# volSteel = 0.1  # Case 1, 10% steel (by volume)
volSteel = 0.9  # Case 2, 90% steel (by volume)

volWater = 1- volSteel

H2O = {'O': O, 'H': H,  'density': volWater * waterDensity }
steel = {'Fe': Fe,'C': carbon, 'Mn': manganese, 'Si': silicon, 'Ni': nickel, 'Cr': chromium, 'Mo': molybdenum, 'density':volSteel * steelDensity }

# input = H2O
input = steel



if __name__=="__main__":

    from nuclearDensityCalculator import results
    file_out = 'nuclearDensityOutputs.txt'
    x = results(input)
    x.display()
    x.write_results(file_out)