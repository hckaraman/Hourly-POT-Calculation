#region modules

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
from matplotlib.offsetbox import AnchoredText
import matplotlib.dates as mdates
pd.plotting.register_matplotlib_converters(explicit=True)
seaborn.set()
np.seterr(all='ignore')

#endregion


class et(object):
    _dir = r'D:\DRIVE\TUBITAK\ET'
    _data = "ET.csv"
    def __init__(self):
        self._working_directory = None
        self.Data_file = None
        self.df = None
        self.R = None
        self.T = None
        self.U = None
        self.Td = None
        self.J = None
        self.t = None
        self.Ett = None
        self.Ets = None
        self.Lat = 38.5
        self.Long = 121.5
        self.Meridian = 120
        self.Elevation = 18.5
        self.sunangle = 17.0
        self.zw = 2.  # ??
        self.Date = None

    @property
    def process_path(self):
        return self._working_directory

    @process_path.setter
    def process_path(self, value):
        self._working_directory = value
        pass

    def DataRead(self):
        self.df = pd.read_csv(self.Data_file, sep=',', parse_dates=[0], header=0)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.set_index('Date')
        self.Date = self.df.index.to_list()
        # self.df['Day'] = pd.DatetimeIndex(self.Date).day
        self.df['dayofyear'] = pd.DatetimeIndex(self.Date).day
        self.df['hour'] = pd.DatetimeIndex(self.Date).hour
        self.df['hour'] = self.df['hour'] + 1



    def InitData(self):
        self.R = self.df.R
        self.T = self.df.Temp
        self.U = self.df.U
        self.Td = self.df.Td
        self.J = np.array(self.df.dayofyear[:])
        self.t = np.array(self.df.hour)
        self.n = self.df.__len__()


    def et_calc(self):
        self.DataRead()
        self.InitData()

        # GSC = solar constant in MJ m-2 min-1
        GSC = 0.082

        # σ = Steffan-Boltzman constant in MJ m-2 h-1 K-4
        SB = 2.04e-10

        # Latitude in radians converted from latitude (L) in degrees
        Teta = math.pi * self.Lat / 180.

        # dr = correction for eccentricity of Earth’s orbit around the sun
        dr = 1 + 0.033 * np.cos(2 * math.pi * self.J / 365.)

        # δ = Declination of the sun above the celestial equator in radians
        Dec = 0.409 * np.sin((2 * math.pi * self.J / 365.) - 1.39)

        # Sc = solar time correction for wobble in Earth’s rotation
        Sc = 0.1645 * np.sin(2 * (2 * math.pi * (self.J - 81) / 364.)) - 0.1255 * np.cos(
            (2 * math.pi * (self.J - 81) / 364.)) - 0.025 * np.sin((2 * math.pi * (self.J - 81) / 364.))

        # ω = hour angle in radians
        # ω1 = hour angle ½ hour before ω in radians
        # ω2 = hour angle ½ hour after ω in radians

        w = (math.pi / 12) * (((self.t - 0.5) + 0.06667 * (self.Meridian - self.Long) + Sc - 12))
        w1 = w - 0.5 * math.pi / 12.
        w2 = w + 0.5 * math.pi / 12.

        # Ra = extraterrestrial radiation (MJ m-2 h-1)

        sint = (w2 - w1) * math.sin(Teta) * np.sin(Dec) + math.cos(Teta) * np.cos(Dec) * (
                np.sin(w2) - np.sin(w1))
        Ra = dr * sint * 60.0 * GSC * 12.0 / (math.pi)

        # β = solar altitude in degrees

        Beta = np.where(Ra < 0, 0,
                        np.arcsin(math.sin(Teta) * np.sin(Dec) + math.cos(Teta) * np.cos(Dec) * np.cos(w)) * 180. / math.pi)
        # Rso = clear sky total global solar radiation at the Earth’s surface in MJ m-2 h-1

        Rso = np.where(Beta == 0, 0, Ra * (0.75 + 2.0e-5 * self.Elevation))

        Rs = np.where(self.R < 0, 0, self.R * 0.0036)

        # es = saturation vapor pressure (kPa) at the mean hourly air temperature (T) in oC
        # ea = actual vapor pressure or saturation vapor pressure (kPa) at the mean dew point temperature
        # ε′ = apparent ‘net’ clear sky emissivity
        # es = np.zeros(n)
        # ea = np.zeros(n)
        # eps = np.zeros(n)

        es = 0.6108 * np.exp(17.27 * self.T / (self.T + 237.3))
        ea = 0.6108 * np.exp(17.27 * self.Td / (self.Td + 237.3))
        eps = 0.34 - 0.14 * np.sqrt(ea)
        ratio = np.zeros(self.n)
        ratio = np.where(Beta < self.sunangle, 0, np.where(Rs / Rso < 0.3, 0.3, np.where(Rs / Rso > 1, 1, Rs / Rso)))

        # f = a cloudiness function of RS and RSO

        f = np.zeros(self.n)
        f[0] = np.where(Beta[0] < self.sunangle, 0.6, 1.35 * ratio[0] - 0.35)
        f[1:self.n] = np.where(Beta[1:self.n] < self.sunangle, f[0:self.n - 1], 1.35 * ratio[1:self.n] - 0.35)

        # Rns = net short wave radiation as a function of measured solar radiation (Rs) in MJ m-2 h-1
        # Rns = np.zeros(n)

        Rns = (1 - 0.23) * Rs

        # Rnl = net long wave radiation in MJ m-2 h-1
        # Rn = net radiation over grass in MJ m-2 h-1

        Rnl = - f * eps * SB * ((self.T + 273.15) ** 4)
        Rn = Rns + Rnl

        # Bp = barometric pressure in kPa as a function of elevation (El) in meters

        Bp = 101.3 * (((293 - 0.0065 * self.Elevation) / 293) ** 5.26)

        # λ = latent heat of vaporization in (MJ kg-1 )

        Alfa = 2.45

        # γ = psychrometric constant in kPao C-1

        psi = np.zeros(self.n)
        psi[:] = 0.00163 * Bp / Alfa

        Gs = np.where(Rn < 0, 0.5 * Rn, 0.1 * Rn)
        Gt = np.where(Rn < 0, 0.2 * Rn, .04 * Rn)

        # wind speed

        u2 = self.U * (4.87 / (math.log(67.8 * self.zw - 5.42)))

        # ra = aerodynamic resistance in s m-1 is estimated for a 0.12 m tall crop as a function of

        rs = np.where(Rn < 0, 200, 50)
        ra = np.where(self.U < 0.5, 208 / 0.5, 208 / self.U)

        # Modified psychrometric constant (γ∗)
        # For short canopy

        Ks = psi * (1 + rs / ra)

        rs = np.where(Rn < 0, 200, 30)
        ra = np.where(self.U < 0.5, 118 / 0.5, 118 / self.U)

        Kt = psi * (1 + rs / ra)

        # ∆ = slope of the saturation vapor pressure curve (kPao C-1 ) at mean air temperature (T)

        delta = 4099. * es / ((self.T + 237.3) ** 2)

        # G = soil heat flux density (MJ m-2 h-1)
        # Gos for ETos
        # Grs for ETrs

        Gos = np.where(Rn > 0, 0.1 * Rn, 0.5 * Rn)
        Grs = np.where(Rn > 0, 0.04 * Rn, 0.2 * Rn)

        # R is the radiation term of the Penman-Monteith and Penman equations in mm d-1 .

        Ros = np.where(Rn > 0, (0.408 * delta * (Rn - Gos) / (delta + psi * (1 + 0.24 * u2))),
                       0.408 * delta * (Rn - Gos) / (delta + psi * (1 + 0.96 * u2)))
        Rot = np.where(Rn > 0, (0.408 * delta * (Rn - Gos) / (delta + psi * (1 + 0.25 * u2))),
                       0.408 * delta * (Rn - Gos) / (delta + psi * (1 + 1.7 * u2)))
        Rop = (0.408 * delta * (Rn - Gos)) / (delta + psi)

        # A = aerodynamic term of the Penman-Monteith equation in mm d-1 with u2 the wind
        # speed at 2 m height
        # As = np.zeros(n)
        # At = np.zeros(n)
        # Ap = np.zeros(n)

        As = np.where(Rn > 0, ((37 * psi / (self.T + 273)) * u2 * (es - ea)) / (delta + psi * (1 + 0.24 * u2)),
                      ((37 * psi / (self.T + 273)) * u2 * (es - ea)) / (delta + psi * (1 + 0.96 * u2)))
        At = np.where(Rn > 0, ((66 * psi / (self.T + 273)) * u2 * (es - ea)) / (delta + psi * (1 + 0.25 * u2)),
                      ((66 * psi / (self.T + 273)) * u2 * (es - ea)) / (delta + psi * (1 + 1.7 * u2)))
        Ap = np.where(Rn > 0, ((37 * psi / (self.T + 273)) * u2 * (es - ea)) / (delta + psi),
                      ((37 * psi / (self.T + 273)) * u2 * (es - ea)) / (delta + psi))

        self.Ets = Ros + As
        self.Ett = Rot + At

        LEs = self.Ets * 2.45
        Hs = Rn - Gs - LEs
        return self.Ets,self.Ett

    def write_topd(self):
        self.df['Short_ET'] = self.Ets
        self.df['Tall_ET'] = self.Ett

    def interpolation(self):
        fit = np.polyfit(self.Ets, self.Ett, 1)
        fit_fn = np.poly1d(fit)
        return fit_fn

    def stats(self,):
        return stats.linregress(self.Ets, self.Ett)

    # Exports results

    def export(self,path):
        self.write_topd()
        self.df.to_csv(os.path.join(path, "Output.csv"),columns=['Short_ET','Tall_ET'])

    def draw(self):
        fit = self.interpolation()
        stats = self.stats()
        fig, ax1 = plt.subplots(figsize = (12,8))
        ax1.set_title('Short and Tall Canopy Comparison', style='italic', fontweight='bold', fontsize=16)
        color = 'tab:orange'
        ax1.set_xlabel(r'Hpurly $Et_s$ (mm $hr^{-1}$)', style='italic', fontweight='bold', fontsize=14)
        ax1.set_ylabel(r'Hourly $Et_t$ (mm $hr^{-1}$)', color=color, style='italic', fontweight='bold', fontsize=14)
        ax1.plot(self.Ets, self.Ett, 'bo', self.Ets, fit(self.Ets), '--k')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x')
        anchored_text = AnchoredText("y = %.2f\n$R^2$ = %0.2f" %(stats[0],(stats[2]) ** 2), loc=5)
        ax1.add_artist(anchored_text)
        fig.tight_layout()
        # plt.show()

    def drawall(self):
        self.write_topd()
        fit = self.interpolation()
        stats = self.stats()
        f = plt.figure(figsize=(12, 8))
        ax1 = f.add_subplot(212)
        ax2 = f.add_subplot(211,sharex=ax1,sharey = ax1)
        # ax3 = f.add_subplot(312)
        color = 'tab:blue'
        ax2.set_ylabel(r'Hpurly $Et_o$ (mm $hr^{-1}$)', color=color, style='italic', fontweight='bold', fontsize=14)
        ax2.plot(self.df['Tall_ET'])
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='x', labelrotation=45)
        ax2.legend(['Tall Canopy Evapotranspiration'])
        ax1.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=13)
        # ax3.set_ylabel(r'Short and Tall Canopy Comparison', color=color, style='italic', fontweight='bold', fontsize=14)
        # ax3.plot(self.Ets, self.Ett, 'bo', self.Ets, fit(self.Ets), '--k')
        # ax3.tick_params(axis='y', labelcolor=color)
        # ax3.tick_params(axis='x', labelrotation=45)
        # ax3.legend(['Tall Canopy Evapotranspiration'])
        # ax3.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=13)
        color = 'tab:orange'
        ax2.set_title('Potentional Evapotranspiration', style='italic', fontweight='bold', fontsize=16)
        ax1.set_ylabel(r'Hourly $Et_o$ (mm $hr^{-1}$)', color=color, style='italic', fontweight='bold', fontsize=14)
        ax1.plot(self.df['Short_ET'],color = color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(['Short Canopy Evapotranspiration'])
        # plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        # plt.setp(ax3.get_xticklabels(), visible=False)
        f.tight_layout()
        self.draw()
        plt.show()

# Initilize object
a = et()
# Process path
a.process_path = r'D:\DRIVE\TUBITAK\ET'
# Data file
a.Data_file = os.path.join(a.process_path, "ET.csv")
# Calculate POT
a.et_calc()
# Export results to a specified path
a.export(r'D:\DRIVE\TUBITAK\ET')
# Draw results
a.drawall()