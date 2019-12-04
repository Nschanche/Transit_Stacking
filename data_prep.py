
def clean_data(df):
	import numpy as np
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.dropna()
	return df

def get_gaia(df):
	import numpy as np
	import pandas as pd
	from astroquery.vizier import Vizier
	Vizier.VIZIER_SERVER = 'vizier.ast.cam.ac.uk'

	v2 = Vizier()
	Gaia_teff = np.zeros(len(df))
	Gaia_rad = np.ones(len(df))
	ii = 0
	for index, row in df.iterrows():
		star_name = index[7:] #Get the J-coordinates
		try:
			tab = v2.query_object(star_name, radius=2*u.arcsec, catalog='Gaia DR2')['I/345/gaia2']
		except:
			tab=[]
		if len(tab) == 1:
			Gaia_teff[ii] = tab['Teff'][0]
			Gaia_rad[ii] = tab['Rad'][0]
		elif len(tab) == 0:
			#print('%s is not in Gaia dr2. Replace with j-h value'%star_name)
			Gaia_rad[ii] = row['rstar_jh']
			Gaia_teff[ii] = row['teff_jh']
		else:
			#print('%s has %s star(s) within 2 arcsecs. Sticking with j-h radius and teff'%(star_name,len(tab)))
			Gaia_rad[ii] = row['rstar_jh']
			Gaia_teff[ii] = row['teff_jh']
		ii+=1
	return(Gaia_rad, Gaia_teff)

def get_trans_ratio(df):
	trans_ratio = (df['npts_intrans']/df['npts_good'])/df['width']
	return trans_ratio

def get_near_int(df):
	near_int = abs(((df['period']+0.5)%1)-0.5)
	return near_int


#Use Gaia radius instead?
def get_rm_ratio(df):
	rm_ratio = df['rstar_mcmc']/df['mstar_mcmc']
	return rm_ratio

def get_depth_to_width(df):
	depth_to_width = df['depth']/df['width']
	return depth_to_width


#There were some problems with the radius and effective temperature measures in the original datafile.
#It looks like they were calculated before updated jh values were given
#re-calculate them with the equations given in the Cameron et al 2007 appendix
def get_temp(df):
	import numpy as np
	teff_jh = np.multiply(-4369.5,df['jmag-hmag']) + 7188.2
	teff_jh = teff_jh.astype(int)
	return teff_jh

#Update the radius estimate based on the revised j-h values
def get_rstar_jh(df):
	rstar_jh = (-3.925e-14*df['teff_jh']**4) + (8.3909e-10*df['teff_jh']**3)- (6.555e-6*df['teff_jh']**2) + (0.02245*df['teff_jh']) - 27.9788
	return rstar_jh

#G
def get_mstar_jh(df):
	mstar_jh = df['rstar_jh']**(1/0.8)
	return mstar_jh

#Find the difference between the stellar mass inferred by color and the mcmc
def get_delta_m(df):
	delta_m = (df['mstar_jh'] - df['mstar_mcmc'])/df['mstar_jh']
	return delta_m

#Find the difference between radius inferred by color and mcmc
def get_delta_r(df):
	delta_r = (df['rstar_jh'] - df['rstar_mcmc'])/df['rstar_jh']
	return delta_r


def get_delta_Gaia(df):
	delta_Gaia = (df['rstar_Gaia'] - df['rstar_mcmc'])/df['rstar_Gaia']
	return delta_Gaia