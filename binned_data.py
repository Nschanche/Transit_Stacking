def get_binned_data(df, nbins=500, plot=False, get_pdgrm=False,get_full_lc=False, get_local_lc=False):
	import pandas as pd
	import os
	import numpy as np
	from astropy.io import fits
	binned_pdgrm = pd.DataFrame()
	binned_lc = pd.DataFrame()
	binned_local = pd.DataFrame()
	dummy_field = '0000_000'
	for index, row in df.iterrows():
		field_name = '%s_%s'%(row['field'],row['camera_id'])
		fname = '/Volumes/Klondike/st_andrews/ORCA_TAMTFA/%s/orion_ORCA_TAMTFA_%s.fits' %(field_name,field_name)
		if os.path.isfile(fname) and field_name != 'OF1127-4254_200' and field_name != dummy_field:
			dummy_field=field_name
			hdulist = fits.open(fname)
			periods = hdulist[6].data #Contains the list of period folds in seconds
			periods = periods[0][0]/(60*60*24) #Converts period to days
            
			candidates = hdulist[4].data
			lightcurves = hdulist[5].data #Contains obs times, magnitudes, and errors.
			periodogram = hdulist[7].data #Contains the calculated periodogram
            
			_list = []
			[_list.append(item[0]) for item in periodogram]
			try:
				want = _list.index(index) #good for PERIODOGRAM
			except ValueError:
				want = -1
				print('%s is not in the fits file' %index)
				print('######################')
				print('scp ns81@ngtshead.warwick.ac.uk:/wasp/lcfiles/orion/ORCA_TAMTFA/%s/orion_ORCA_TAMTFA_%s.fits /Volumes/Klondike/st_andrews/ORCA_TAMTFA/%s/'%(field_name,field_name,field_name))
				print('######################')
			name = periodogram[want]['obj_id']
			chisq = periodogram[want]['chisq']
            
            
			_list2 = []
			[_list2.append(item[0]) for item in candidates]

        #if want != -1:
			try:
				want2 = _list2.index(index) #good for CANDIDATES and LIGHTCURVES
               
			except ValueError:
				want2 = -1
				print('%s is not in the fits file' %index)
				print('######################')
				print('scp ns81@ngtshead.warwick.ac.uk:/wasp/lcfiles/orion/ORCA_TAMTFA/%s/orion_ORCA_TAMTFA_%s.fits /Volumes/Klondike/st_andrews/ORCA_TAMTFA/%s/'%(field_name,field_name,field_name))
				print('######################')
			#print(candidates[want2]['obj_id'])
			per = candidates[want2]['period'] #period in seconds
			width = candidates[want2]['width'] #width in seconds
			epoch = candidates[want2]['epoch']
			hjd = np.array(lightcurves[want2]['hjd']) #-epoch if you want it centered
			mag = (lightcurves[want2]['mag'])*-1
			#print(name,per,width,epoch)

			if get_pdgrm:
				binned_pdgrm[name] = get_binned_periodograms(index, periods, chisq, plot=plot)
			if get_full_lc:
				binned_lc[name] = get_binned_lc(hjd, mag, per, name, plot=plot)
			if get_local_lc==True:
				binned_local[name] = get_local_binned_lc(hjd-epoch, mag, per, name, width, plot=plot)
	binned_pdgrm = binned_pdgrm.transpose()
	binned_lc = binned_lc.transpose()
	binned_local = binned_local.transpose()
	return(binned_pdgrm, binned_lc, binned_local)

def get_binned_periodograms(name, periods, chisq, nbins=500, plot=False):
	import numpy as np
	from matplotlib import pyplot as plt
	actual_bins = np.linspace(min(periods), max(periods), nbins+1)
	#bin_periods = np.zeros([nbins])
	bin_pdgrm = np.zeros([nbins])
	for ii in range(nbins):
		zoom = np.ma.masked_inside(periods, actual_bins[ii], actual_bins[ii+1])
		bin_pdgrm[ii] = np.min(chisq[zoom.mask])
	if plot:
		#print min(bin_chisq)
		plt.figure(figsize=(14,6)) 
		plt.title('%s'%name, fontsize=20)
		#plt.xlim([np.log(.35),np.log(16)])
		#plt.xlim([0,16])
		#plt.axvline(x=np.log(0.5*per), linestyle='--', color='darkgray', lw=2)
		#plt.axvline(x=np.log(per), linestyle='--', color='red', lw=2)
		#plt.axvline(x=np.log(2*per), linestyle='--', color='darkgray', lw=2)
		#plt.axvline(x=np.log(3*per), linestyle='--', color='darkgray', lw=2)
		#plt.plot(bin_periods, bin_chisq)
		#plt.plot(np.log(bin_periods), bin_chisq)
		plt.plot(bin_pdgrm)
		#plt.xlabel("Period (log days)", fontsize=15)
		plt.ylabel("$\Delta\chi^2$", fontsize=15)
		#plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		#plt.savefig('1SWASP_J134422_58+480143.2_unbinnedlc.png')
		plt.show()
                
                
            
		#hdulist.close()
	#binned_periodogram[name] = bin_chisq
	return bin_pdgrm



def get_binned_lc(hjd, mag, per, name, nbins=500, plot=False):
	from matplotlib import pyplot as plt
	import numpy as np
	phase_time = (hjd/per)%1
	avg_mag = np.zeros(nbins)
	avg_phase = np.zeros(nbins)
	for ii in range(int(nbins)):  
		phases = np.nonzero((phase_time >= float(ii)/nbins) & (phase_time < (ii+1.)/nbins))
        #print(phases[0])
		if len(phases[0]) == 0:
			avg_mag[ii] = 0.1
			avg_phase[ii] = float(ii+.5)/nbins #.5
		else:
			avg_mag[ii] = np.mean(mag[phases[0]])
			avg_phase[ii] = np.mean(phase_time[phases[0]])
		
        #avg_phase[avg_phase > .5] = avg_phase[avg_phase > .5] - 1
        #sorted_phase = np.argsort(avg_phase)
        #avg_phase = avg_phase[sorted_phase]
        #avg_mag = avg_mag[sorted_phase]

	if plot:
		plt.figure(figsize=(8,6))
		plt.title('%s'%name, fontsize=20)   
		plt.ylim([-.15,.15])
		plt.plot(avg_mag)
		plt.ylabel("Delta Mag", fontsize=15)
		plt.yticks(fontsize=12)
		#plt.savefig('1SWASP_J134422_58+480143.2_unbinnedlc.png')
		plt.show()
            
    #hdulist.close()
	return avg_mag


def get_local_binned_lc(hjd, mag, per, name, width, nbins=250, plot=False):
	import numpy as np
	from matplotlib import pyplot as plt
	phase_time = (hjd/per)%1
    #Now let's find the local view
	local_binmag = np.zeros(nbins)
	local_phase = phase_time[np.nonzero((phase_time >= (1. - (width/per)*2)) | (phase_time <= (0. + (width/per)*2)))]
	local_phase[local_phase > 0.5] = local_phase[local_phase > 0.5] - 1 
	local_mag = mag[np.nonzero((phase_time >= (1. - (width/per)*2)) | (phase_time <= (0. + (width/per)*2)))]
	actual_bins = np.linspace(min(local_phase), max(local_phase), nbins+1)

	temp = 0
	no_data_count = 0
	for ii in range(nbins):
		phases = local_mag[(local_phase >= actual_bins[ii]) & (local_phase < actual_bins[ii+1])]

		if ii==0:
			if len(phases) == 0:
				local_binmag[ii] = .1 #local_binmag[temp-1]
				no_data_count = no_data_count+1
			else:
				local_binmag[ii] = np.mean(phases)
		else:
			local_binmag[ii] = np.mean(phases)


    #print('There were %s bins that had no data'%no_data_count)

	if plot == True:
		name = name.replace(" ","")
		name = name.replace(".","_") 
		plt.figure(figsize=(12,6))
		plt.title('%s'%name, fontsize=20)
		plt.ylim([-.05,.05])
		plt.plot(local_binmag)
		plt.ylabel("Delta Mag", fontsize=15)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		#plt.savefig('%s_unbinnedlc.png'%name)
		plt.show()
        
	#hdulist.close()
   
	return local_binmag
