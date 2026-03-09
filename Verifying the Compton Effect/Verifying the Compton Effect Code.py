import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import trapz

def get_rdd_chi_sqrd_val(obs_array,calc_array,sigma_vals_array,num_free_params):
    "RETURNS: Reduced Chi-squared value for fitted gaussian."
    chi_sqrd_list = []
    num_obs = len(obs_array)
    obs_list = list(obs_array)
    calc_list = list(calc_array)
    sigma_vals_list = list(sigma_vals_array)
    for i in range(0,len(obs_list)):
        if sigma_vals_list[i] == 0:
            sigma_vals_list[i] = 1
        elif sigma_vals_list[i] != 0:
            sigma_vals_list[i] = sigma_vals_list[i]
        chi_sqrd_val = ((obs_list[i] - calc_list[i])**2) / (sigma_vals_list[i]**2)
        chi_sqrd_list.append(chi_sqrd_val)
    chi_sqrd = np.sum(chi_sqrd_list)
    rdd_chi_sqrd = chi_sqrd / (num_obs - num_free_params)
    return rdd_chi_sqrd
#------------------------Importing measured data:------------------------------
#We want to import all scattering angle data:
#Bin Numbers Array for ALL Angle Measurements:
n_vals = np.loadtxt( # Bin Number, n
    "Verifying the Compton Effect Data/Set 1 deg 0 data.csv", delimiter=',', usecols=0,
    skiprows=1)

#Gaining Total Count Numbers for EACH Angle measurement:
deg_0_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 0 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_10_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 10 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_20_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 20 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_30_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 30 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_40_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 40 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_50_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 50 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_60_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 60 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_70_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 70 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_80_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 80 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_90_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 90 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_100_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 100 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_110_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 110 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_120_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 120 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_130_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 130 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_140_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 140 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#------------------------------------------------------------------------------
deg_150_N_vals = np.loadtxt( # Total Count Number, N, WITH SCATTERING TARGET.
    "Verifying the Compton Effect Data/Set 1 deg 150 data.csv", delimiter=',', usecols=1,
    skiprows=1)
#%%----------------------------------------------------------------------------
#-------------------------------Plotting:--------------------------------------
#Want to gain Gaussian Fit and Standard Deviations:
def fit_gauss_lin_func(n_domain, amp, mu, sigma, const, m_bg):
    '''Returns Array of Fitted N-values.'''
    gaus = amp*np.exp(-(n_domain-mu)**2 / (2 * sigma**2)) + const + m_bg*n_domain
    return gaus

def fit_gauss_func(n_domain, amp, mu, sigma):
    '''Returns Array of Fitted N-values.'''
    gaus = amp*np.exp(-(n_domain-mu)**2 / (2 * sigma**2))
    return gaus

def get_sqr_rt_N_vals_fit(N_vals_fit_array):
    "RETURNS: Array of the square root of abs vals of N fit array."
    return_sqr_rt_N_vals_fit_list = []
    N_vals_fit_list = list(N_vals_fit_array)
    for N_val in N_vals_fit_list:
        sqr_rt_N_val = np.sqrt(np.maximum(N_val, 1)) #CHANGED FOR ANALYSIS
        return_sqr_rt_N_vals_fit_list.append(sqr_rt_N_val)
    return_sqr_rt_N_vals_fit_array = np.array(return_sqr_rt_N_vals_fit_list)
    return return_sqr_rt_N_vals_fit_array
    
#------------------------------------------------------------------------------
#0 degs:
plt.plot(n_vals,deg_0_N_vals, color='red', marker=".", linestyle='',
          label="0 degs")

deg_0_n_fit_domain = n_vals[75:116]

deg_0_init_guesses = [3_500, 90, 3, 0, -12]

fit_deg_0,cov_deg_0 = curve_fit(fit_gauss_lin_func, deg_0_n_fit_domain,
                                deg_0_N_vals[75:116], deg_0_init_guesses)

deg_0_mu_err = np.sqrt(cov_deg_0[1,1])  #The uncertainty in the mean.
deg_0_sig_err = np.sqrt(cov_deg_0[2,2]) #The uncertainty in the std deviation.

print('----------------------------------------------------------------------')
print('0 Degrees Gaussian fit coefficients:')
print(fit_deg_0)
print('\n0 Degrees Covariance matrix:')
print(cov_deg_0)
print('\n0 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_0[1]} +/- {deg_0_mu_err}')
print('\n0 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_0[2]} +/- {deg_0_sig_err}')

deg_0_gauss_plot_domain = np.linspace(75, 115, 10_000)
deg_0_gauss_plot_vals = fit_gauss_lin_func(deg_0_gauss_plot_domain, amp=fit_deg_0[0],
                                        mu=fit_deg_0[1], sigma=fit_deg_0[2],
                                        const=fit_deg_0[3], m_bg=fit_deg_0[4])

plt.plot(deg_0_gauss_plot_domain, deg_0_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
plt.yticks(np.arange(0, 4_000+500, 500))
plt.xlim(0,150)
plt.ylim(0,4_000)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 0 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#10 degs:
plt.plot(n_vals,deg_10_N_vals, color='red', marker=".", linestyle='',
          label="10 degs")

deg_10_n_fit_domain_start = 80
deg_10_n_fit_domain_end = 100

deg_10_n_fit_domain = n_vals[deg_10_n_fit_domain_start:deg_10_n_fit_domain_end]

deg_10_init_guesses = [5000, 91, 2, 0, -20] #amp,mu,std dev,const,grad

deg_10_N_vals_fit = deg_10_N_vals[deg_10_n_fit_domain_start:deg_10_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_10_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_10_N_vals_fit)

fit_deg_10,cov_deg_10 = curve_fit(fit_gauss_lin_func, deg_10_n_fit_domain,
                                deg_10_N_vals_fit, deg_10_init_guesses)

deg_10_mu_err = np.sqrt(cov_deg_10[1,1])  #The uncertainty in the mean.
deg_10_sig_err = np.sqrt(cov_deg_10[2,2]) #The uncertainty in the std deviation.

deg_10_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_10_N_vals_fit,
                                                fit_gauss_lin_func(deg_10_n_fit_domain,
                                                                   fit_deg_10[0],
                                                                   fit_deg_10[1],
                                                                   fit_deg_10[2],
                                                                   fit_deg_10[3],
                                                                   fit_deg_10[4]),
                                                deg_10_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('10 Degrees Gaussian fit coefficients:')
print(fit_deg_10)
print('\n10 Degrees Covariance matrix:')
print(cov_deg_10)
print('\n10 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_10[1]} +/- {deg_10_mu_err}')
print('\n10 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_10[2]} +/- {deg_10_sig_err}')
print('\n10 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_10_rdd_chi_sqrd_val}')

deg_10_gauss_plot_domain = np.linspace(deg_10_n_fit_domain_start,
                                        deg_10_n_fit_domain_end-1,
                                        10_000)
deg_10_gauss_plot_vals = fit_gauss_lin_func(deg_10_gauss_plot_domain, amp=fit_deg_10[0],
                                        mu=fit_deg_10[1], sigma=fit_deg_10[2],
                                        const=fit_deg_10[3], m_bg=fit_deg_10[4])

plt.plot(deg_10_gauss_plot_domain, deg_10_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
plt.yticks(np.arange(0, 5_000+500, 500))
plt.xlim(0,150)
plt.ylim(0,5_000)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 10 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#20 degs:
plt.plot(n_vals,deg_20_N_vals, color='red', marker=".", linestyle='',
          label="20 degs")

deg_20_n_fit_domain_start = 68
deg_20_n_fit_domain_end = 101

deg_20_n_fit_domain = n_vals[deg_20_n_fit_domain_start:deg_20_n_fit_domain_end]

deg_20_init_guesses = [50, 84, 5, 0, -3] #amp,mu,std dev,const,grad

deg_20_N_vals_fit = deg_20_N_vals[deg_20_n_fit_domain_start:deg_20_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_20_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_20_N_vals_fit)

fit_deg_20,cov_deg_20 = curve_fit(fit_gauss_lin_func, deg_20_n_fit_domain,
                                deg_20_N_vals_fit, deg_20_init_guesses)

deg_20_mu_err = np.sqrt(cov_deg_20[1,1])  #The uncertainty in the mean.
deg_20_sig_err = np.sqrt(cov_deg_20[2,2]) #The uncertainty in the std deviation.

deg_20_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_20_N_vals_fit,
                                                fit_gauss_lin_func(deg_20_n_fit_domain,
                                                                   fit_deg_20[0],
                                                                   fit_deg_20[1],
                                                                   fit_deg_20[2],
                                                                   fit_deg_20[3],
                                                                   fit_deg_20[4]),
                                                deg_20_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('20 Degrees Gaussian fit coefficients:')
print(fit_deg_20)
print('\n20 Degrees Covariance matrix:')
print(cov_deg_20)
print('\n20 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_20[1]} +/- {deg_20_mu_err}')
print('\n20 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_20[2]} +/- {deg_20_sig_err}')
print('\n20 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_20_rdd_chi_sqrd_val}')

deg_20_gauss_plot_domain = np.linspace(deg_20_n_fit_domain_start,
                                        deg_20_n_fit_domain_end-1,
                                        10_000)
deg_20_gauss_plot_vals = fit_gauss_lin_func(deg_20_gauss_plot_domain, amp=fit_deg_20[0],
                                        mu=fit_deg_20[1], sigma=fit_deg_20[2],
                                        const=fit_deg_20[3], m_bg=fit_deg_20[4])

plt.plot(deg_20_gauss_plot_domain, deg_20_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 55+5, 5))
plt.xlim(0,150)
#plt.ylim(0,55)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 20 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#30 degs:
plt.plot(n_vals,deg_30_N_vals, color='red', marker=".", linestyle='',
          label="30 degs")

deg_30_n_fit_domain_start = 65
deg_30_n_fit_domain_end = 85

deg_30_n_fit_domain = n_vals[deg_30_n_fit_domain_start:deg_30_n_fit_domain_end]

deg_30_init_guesses = [65, 77, 3, 45, -0.2] #amp,mu,std dev,const,grad

deg_30_N_vals_fit = deg_30_N_vals[deg_30_n_fit_domain_start:deg_30_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_30_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_30_N_vals_fit)

fit_deg_30,cov_deg_30 = curve_fit(fit_gauss_lin_func, deg_30_n_fit_domain,
                                deg_30_N_vals_fit, deg_30_init_guesses)

deg_30_mu_err = np.sqrt(cov_deg_30[1,1])  #The uncertainty in the mean.
deg_30_sig_err = np.sqrt(cov_deg_30[2,2]) #The uncertainty in the std deviation.

deg_30_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_30_N_vals_fit,
                                                fit_gauss_lin_func(deg_30_n_fit_domain,
                                                                   fit_deg_30[0],
                                                                   fit_deg_30[1],
                                                                   fit_deg_30[2],
                                                                   fit_deg_30[3],
                                                                   fit_deg_30[4]),
                                                deg_30_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('30 Degrees Gaussian fit coefficients:')
print(fit_deg_30)
print('\n30 Degrees Covariance matrix:')
print(cov_deg_30)
print('\n30 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_30[1]} +/- {deg_30_mu_err}')
print('\n30 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_30[2]} +/- {deg_30_sig_err}')
print('\n30 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_30_rdd_chi_sqrd_val}')

deg_30_gauss_plot_domain = np.linspace(deg_30_n_fit_domain_start,
                                        deg_30_n_fit_domain_end-1,
                                        10_000)
deg_30_gauss_plot_vals = fit_gauss_lin_func(deg_30_gauss_plot_domain, amp=fit_deg_30[0],
                                        mu=fit_deg_30[1], sigma=fit_deg_30[2],
                                        const=fit_deg_30[3], m_bg=fit_deg_30[4])

plt.plot(deg_30_gauss_plot_domain, deg_30_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 40+5, 5))
plt.xlim(0,150)
#plt.ylim(0,40)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 30 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#40 degs:
plt.plot(n_vals,deg_40_N_vals, color='red', marker=".", linestyle='',
          label="40 degs")

deg_40_n_fit_domain_start = 50
deg_40_n_fit_domain_end = 80

deg_40_n_fit_domain = n_vals[deg_40_n_fit_domain_start:deg_40_n_fit_domain_end]

deg_40_init_guesses = [50, 70, 5, 75, -1] #amp,mu,std dev,const,grad

deg_40_N_vals_fit = deg_40_N_vals[deg_40_n_fit_domain_start:deg_40_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_40_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_40_N_vals_fit)

fit_deg_40,cov_deg_40 = curve_fit(fit_gauss_lin_func, deg_40_n_fit_domain,
                                deg_40_N_vals_fit, deg_40_init_guesses)

deg_40_mu_err = np.sqrt(cov_deg_40[1,1])  #The uncertainty in the mean.
deg_40_sig_err = np.sqrt(cov_deg_40[2,2]) #The uncertainty in the std deviation.

deg_40_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_40_N_vals_fit,
                                                fit_gauss_lin_func(deg_40_n_fit_domain,
                                                                   fit_deg_40[0],
                                                                   fit_deg_40[1],
                                                                   fit_deg_40[2],
                                                                   fit_deg_40[3],
                                                                   fit_deg_40[4]),
                                                deg_40_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('40 Degrees Gaussian fit coefficients:')
print(fit_deg_40)
print('\n40 Degrees Covariance matrix:')
print(cov_deg_40)
print('\n40 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_40[1]} +/- {deg_40_mu_err}')
print('\n40 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_40[2]} +/- {deg_40_sig_err}')
print('\n40 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_40_rdd_chi_sqrd_val}')

deg_40_gauss_plot_domain = np.linspace(deg_40_n_fit_domain_start,
                                        deg_40_n_fit_domain_end-1,
                                        10_000)
deg_40_gauss_plot_vals = fit_gauss_lin_func(deg_40_gauss_plot_domain, amp=fit_deg_40[0],
                                        mu=fit_deg_40[1], sigma=fit_deg_40[2],
                                        const=fit_deg_40[3], m_bg=fit_deg_40[4])

plt.plot(deg_40_gauss_plot_domain, deg_40_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 50+5, 5))
plt.xlim(0,150)
#plt.ylim(0,50)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 40 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#50 degs:
plt.plot(n_vals,deg_50_N_vals, color='red', marker=".", linestyle='',
          label="50 degs")

deg_50_n_fit_domain_start = 25
deg_50_n_fit_domain_end = 80

deg_50_n_fit_domain = n_vals[deg_50_n_fit_domain_start:deg_50_n_fit_domain_end]

deg_50_init_guesses = [75, 63, 7, 100, -1] #amp,mu,std dev,const,grad

deg_50_N_vals_fit = deg_50_N_vals[deg_50_n_fit_domain_start:deg_50_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_50_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_50_N_vals_fit)

fit_deg_50,cov_deg_50 = curve_fit(fit_gauss_lin_func, deg_50_n_fit_domain,
                                deg_50_N_vals_fit, deg_50_init_guesses)

deg_50_mu_err = np.sqrt(cov_deg_50[1,1])  #The uncertainty in the mean.
deg_50_sig_err = np.sqrt(cov_deg_50[2,2]) #The uncertainty in the std deviation.

deg_50_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_50_N_vals_fit,
                                                fit_gauss_lin_func(deg_50_n_fit_domain,
                                                                   fit_deg_50[0],
                                                                   fit_deg_50[1],
                                                                   fit_deg_50[2],
                                                                   fit_deg_50[3],
                                                                   fit_deg_50[4]),
                                                deg_50_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('50 Degrees Gaussian fit coefficients:')
print(fit_deg_50)
print('\n50 Degrees Covariance matrix:')
print(cov_deg_50)
print('\n50 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_50[1]} +/- {deg_50_mu_err}')
print('\n50 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_50[2]} +/- {deg_50_sig_err}')
print('\n50 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_50_rdd_chi_sqrd_val}')

deg_50_gauss_plot_domain = np.linspace(deg_50_n_fit_domain_start,
                                        deg_50_n_fit_domain_end-1,
                                        10_000)
deg_50_gauss_plot_vals = fit_gauss_lin_func(deg_50_gauss_plot_domain, amp=fit_deg_50[0],
                                        mu=fit_deg_50[1], sigma=fit_deg_50[2],
                                        const=fit_deg_50[3], m_bg=fit_deg_50[4])

plt.plot(deg_50_gauss_plot_domain, deg_50_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 50+5, 5))
plt.xlim(0,150)
#plt.ylim(0,50)
plt.xlabel("Bin Number, n")
plt.ylabel("Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 50 Degrees Gaussian Counts Plot",dpi=600, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#60 degs:
plt.plot(n_vals,deg_60_N_vals, color='red', marker=".", linestyle='',
          label="60 degs")

deg_60_n_fit_domain_start = 49
deg_60_n_fit_domain_end = 65

deg_60_n_fit_domain = n_vals[deg_60_n_fit_domain_start:deg_60_n_fit_domain_end]

deg_60_init_guesses = [62, 56, 7, 100, -1] #amp,mu,std dev,const,grad

deg_60_N_vals_fit = deg_60_N_vals[deg_60_n_fit_domain_start:deg_60_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_60_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_60_N_vals_fit)

fit_deg_60,cov_deg_60 = curve_fit(fit_gauss_lin_func, deg_60_n_fit_domain,
                                deg_60_N_vals_fit, deg_60_init_guesses)

deg_60_mu_err = np.sqrt(cov_deg_60[1,1])  #The uncertainty in the mean.
deg_60_sig_err = np.sqrt(cov_deg_60[2,2]) #The uncertainty in the std deviation.

deg_60_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_60_N_vals_fit,
                                                fit_gauss_lin_func(deg_60_n_fit_domain,
                                                                   fit_deg_60[0],
                                                                   fit_deg_60[1],
                                                                   fit_deg_60[2],
                                                                   fit_deg_60[3],
                                                                   fit_deg_60[4]),
                                                deg_60_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('60 Degrees Gaussian fit coefficients:')
print(fit_deg_60)
print('\n60 Degrees Covariance matrix:')
print(cov_deg_60)
print('\n60 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_60[1]} +/- {deg_60_mu_err}')
print('\n60 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_60[2]} +/- {deg_60_sig_err}')
print('\n60 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_60_rdd_chi_sqrd_val}')

deg_60_gauss_plot_domain = np.linspace(deg_60_n_fit_domain_start,
                                        deg_60_n_fit_domain_end-1,
                                        10_000)
deg_60_gauss_plot_vals = fit_gauss_lin_func(deg_60_gauss_plot_domain, amp=fit_deg_60[0],
                                        mu=fit_deg_60[1], sigma=fit_deg_60[2],
                                        const=fit_deg_60[3], m_bg=fit_deg_60[4])

plt.plot(deg_60_gauss_plot_domain, deg_60_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 40+5, 5))
plt.xlim(0,150)
#plt.ylim(0,40)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 60 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#70 degs:
plt.plot(n_vals,deg_70_N_vals, color='red', marker=".", linestyle='',
          label="70 degs")

deg_70_n_fit_domain_start = 20
deg_70_n_fit_domain_end = 70

deg_70_n_fit_domain = n_vals[deg_70_n_fit_domain_start:deg_70_n_fit_domain_end]

deg_70_init_guesses = [60, 50, 3, 100, -1] #amp,mu,std dev,const,grad

deg_70_N_vals_fit = deg_70_N_vals[deg_70_n_fit_domain_start:deg_70_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_70_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_70_N_vals_fit)

fit_deg_70,cov_deg_70 = curve_fit(fit_gauss_lin_func, deg_70_n_fit_domain,
                                deg_70_N_vals_fit, deg_70_init_guesses)

deg_70_mu_err = np.sqrt(cov_deg_70[1,1])  #The uncertainty in the mean.
deg_70_sig_err = np.sqrt(cov_deg_70[2,2]) #The uncertainty in the std deviation.

deg_70_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_70_N_vals_fit,
                                                fit_gauss_lin_func(deg_70_n_fit_domain,
                                                                   fit_deg_70[0],
                                                                   fit_deg_70[1],
                                                                   fit_deg_70[2],
                                                                   fit_deg_70[3],
                                                                   fit_deg_70[4]),
                                                deg_70_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('70 Degrees Gaussian fit coefficients:')
print(fit_deg_70)
print('\n70 Degrees Covariance matrix:')
print(cov_deg_70)
print('\n70 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_70[1]} +/- {deg_70_mu_err}')
print('\n70 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_70[2]} +/- {deg_70_sig_err}')
print('\n70 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_70_rdd_chi_sqrd_val}')

deg_70_gauss_plot_domain = np.linspace(deg_70_n_fit_domain_start,
                                        deg_70_n_fit_domain_end-1,
                                        10_000)
deg_70_gauss_plot_vals = fit_gauss_lin_func(deg_70_gauss_plot_domain, amp=fit_deg_70[0],
                                        mu=fit_deg_70[1], sigma=fit_deg_70[2],
                                        const=fit_deg_70[3], m_bg=fit_deg_70[4])

plt.plot(deg_70_gauss_plot_domain, deg_70_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 40+5, 5))
plt.xlim(0,150)
#plt.ylim(0,40)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 70 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#80 degs:
plt.plot(n_vals,deg_80_N_vals, color='red', marker=".", linestyle='',
          label="80 degs")

deg_80_n_fit_domain_start = 25
deg_80_n_fit_domain_end = 55

deg_80_n_fit_domain = n_vals[deg_80_n_fit_domain_start:deg_80_n_fit_domain_end]

deg_80_init_guesses = [70, 45, 5, 50, -1] #amp,mu,std dev,const,grad

deg_80_N_vals_fit = deg_80_N_vals[deg_80_n_fit_domain_start:deg_80_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_80_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_80_N_vals_fit)

fit_deg_80,cov_deg_80 = curve_fit(fit_gauss_lin_func, deg_80_n_fit_domain,
                                deg_80_N_vals_fit, deg_80_init_guesses)

deg_80_mu_err = np.sqrt(cov_deg_80[1,1])  #The uncertainty in the mean.
deg_80_sig_err = np.sqrt(cov_deg_80[2,2]) #The uncertainty in the std deviation.

deg_80_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_80_N_vals_fit,
                                                fit_gauss_lin_func(deg_80_n_fit_domain,
                                                                   fit_deg_80[0],
                                                                   fit_deg_80[1],
                                                                   fit_deg_80[2],
                                                                   fit_deg_80[3],
                                                                   fit_deg_80[4]),
                                                deg_80_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('80 Degrees Gaussian fit coefficients:')
print(fit_deg_80)
print('\n80 Degrees Covariance matrix:')
print(cov_deg_80)
print('\n80 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_80[1]} +/- {deg_80_mu_err}')
print('\n80 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_80[2]} +/- {deg_80_sig_err}')
print('\n80 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_80_rdd_chi_sqrd_val}')

deg_80_gauss_plot_domain = np.linspace(deg_80_n_fit_domain_start,
                                        deg_80_n_fit_domain_end-1,
                                        10_000)
deg_80_gauss_plot_vals = fit_gauss_lin_func(deg_80_gauss_plot_domain, amp=fit_deg_80[0],
                                        mu=fit_deg_80[1], sigma=fit_deg_80[2],
                                        const=fit_deg_80[3], m_bg=fit_deg_80[4])

plt.plot(deg_80_gauss_plot_domain, deg_80_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 50+5, 5))
plt.xlim(0,150)
#plt.ylim(0,50)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 80 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#90 degs:
plt.plot(n_vals,deg_90_N_vals, color='red', marker=".", linestyle='',
          label="90 degs")

deg_90_n_fit_domain_start = 20
deg_90_n_fit_domain_end = 60

deg_90_n_fit_domain = n_vals[deg_90_n_fit_domain_start:deg_90_n_fit_domain_end]

deg_90_init_guesses = [110, 41, 7, 150, -4] #amp,mu,std dev,const,grad

deg_90_N_vals_fit = deg_90_N_vals[deg_90_n_fit_domain_start:deg_90_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_90_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_90_N_vals_fit)

fit_deg_90,cov_deg_90 = curve_fit(fit_gauss_lin_func, deg_90_n_fit_domain,
                                deg_90_N_vals_fit, deg_90_init_guesses)

deg_90_mu_err = np.sqrt(cov_deg_90[1,1])  #The uncertainty in the mean.
deg_90_sig_err = np.sqrt(cov_deg_90[2,2]) #The uncertainty in the std deviation.

deg_90_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_90_N_vals_fit,
                                                fit_gauss_lin_func(deg_90_n_fit_domain,
                                                                   fit_deg_90[0],
                                                                   fit_deg_90[1],
                                                                   fit_deg_90[2],
                                                                   fit_deg_90[3],
                                                                   fit_deg_90[4]),
                                                deg_90_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('90 Degrees Gaussian fit coefficients:')
print(fit_deg_90)
print('\n90 Degrees Covariance matrix:')
print(cov_deg_90)
print('\n90 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_90[1]} +/- {deg_90_mu_err}')
print('\n90 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_90[2]} +/- {deg_90_sig_err}')
print('\n90 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_90_rdd_chi_sqrd_val}')

deg_90_gauss_plot_domain = np.linspace(deg_90_n_fit_domain_start,
                                        deg_90_n_fit_domain_end-1,
                                        10_000)
deg_90_gauss_plot_vals = fit_gauss_lin_func(deg_90_gauss_plot_domain, amp=fit_deg_90[0],
                                        mu=fit_deg_90[1], sigma=fit_deg_90[2],
                                        const=fit_deg_90[3], m_bg=fit_deg_90[4])

plt.plot(deg_90_gauss_plot_domain, deg_90_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(10, 150+10, 10))
#plt.yticks(np.arange(0, 60+5, 5))
plt.xlim(0,150)
#plt.ylim(0,60)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 90 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#100 degs:
plt.plot(n_vals,deg_100_N_vals, color='red', marker=".", linestyle='',
          label="100 degs")

deg_100_n_fit_domain_start = 20
deg_100_n_fit_domain_end = 45

deg_100_n_fit_domain = n_vals[deg_100_n_fit_domain_start:deg_100_n_fit_domain_end]

deg_100_init_guesses = [200, 37, 7, 200, -4] #amp,mu,std dev,const,m_bg

deg_100_N_vals_fit = deg_100_N_vals[deg_100_n_fit_domain_start:deg_100_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_100_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_100_N_vals_fit)

fit_deg_100,cov_deg_100 = curve_fit(fit_gauss_lin_func, deg_100_n_fit_domain,
                                deg_100_N_vals_fit, deg_100_init_guesses)

deg_100_mu_err = np.sqrt(cov_deg_100[1,1])  #The uncertainty in the mean.
deg_100_sig_err = np.sqrt(cov_deg_100[2,2]) #The uncertainty in the std deviation.

deg_100_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_100_N_vals_fit,
                                                fit_gauss_lin_func(deg_100_n_fit_domain,
                                                                   fit_deg_100[0],
                                                                   fit_deg_100[1],
                                                                   fit_deg_100[2],
                                                                   fit_deg_100[3],
                                                                   fit_deg_100[4]),
                                                deg_100_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('100 Degrees Gaussian fit coefficients:')
print(fit_deg_100)
print('\n100 Degrees Covariance matrix:')
print(cov_deg_100)
print('\n100 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_100[1]} +/- {deg_100_mu_err}')
print('\n100 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_100[2]} +/- {deg_100_sig_err}')
print('\n100 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_100_rdd_chi_sqrd_val}')

deg_100_gauss_plot_domain = np.linspace(deg_100_n_fit_domain_start,
                                        deg_100_n_fit_domain_end-1,
                                        10_000)
deg_100_gauss_plot_vals = fit_gauss_lin_func(deg_100_gauss_plot_domain, amp=fit_deg_100[0],
                                        mu=fit_deg_100[1], sigma=fit_deg_100[2],
                                        const=fit_deg_100[3], m_bg=fit_deg_100[4])

plt.plot(deg_100_gauss_plot_domain, deg_100_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 80+5, 5))
plt.xlim(0,150)
#plt.ylim(0,80)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 100 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#110 degs:
plt.plot(n_vals,deg_110_N_vals, color='red', marker=".", linestyle='',
          label="110 degs")

deg_110_n_fit_domain_start = 27
deg_110_n_fit_domain_end = 40

deg_110_n_fit_domain = n_vals[deg_110_n_fit_domain_start:deg_110_n_fit_domain_end]

deg_110_init_guesses = [225, 34, 7, 200, -4] #amp,mu,std dev,const,m_bg

deg_110_N_vals_fit = deg_110_N_vals[deg_110_n_fit_domain_start:deg_110_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_110_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_110_N_vals_fit)

fit_deg_110,cov_deg_110 = curve_fit(fit_gauss_lin_func, deg_110_n_fit_domain,
                                deg_110_N_vals_fit, deg_110_init_guesses)

deg_110_mu_err = np.sqrt(cov_deg_110[1,1])  #The uncertainty in the mean.
deg_110_sig_err = np.sqrt(cov_deg_110[2,2]) #The uncertainty in the std deviation.

deg_110_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_110_N_vals_fit,
                                                fit_gauss_lin_func(deg_110_n_fit_domain,
                                                                   fit_deg_110[0],
                                                                   fit_deg_110[1],
                                                                   fit_deg_110[2],
                                                                   fit_deg_110[3],
                                                                   fit_deg_110[4]),
                                                deg_110_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('110 Degrees Gaussian fit coefficients:')
print(fit_deg_110)
print('\n110 Degrees Covariance matrix:')
print(cov_deg_110)
print('\n110 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_110[1]} +/- {deg_110_mu_err}')
print('\n110 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_110[2]} +/- {deg_110_sig_err}')
print('\n110 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_110_rdd_chi_sqrd_val}')

deg_110_gauss_plot_domain = np.linspace(deg_110_n_fit_domain_start,
                                        deg_110_n_fit_domain_end-1,
                                        10_000)
deg_110_gauss_plot_vals = fit_gauss_lin_func(deg_110_gauss_plot_domain, amp=fit_deg_110[0],
                                        mu=fit_deg_110[1], sigma=fit_deg_110[2],
                                        const=fit_deg_110[3], m_bg=fit_deg_110[4])

plt.plot(deg_110_gauss_plot_domain, deg_110_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 130+10, 10))
plt.xlim(0,150)
#plt.ylim(0,130)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 110 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#120 degs:
plt.plot(n_vals,deg_120_N_vals, color='red', marker=".", linestyle='',
          label="120 degs")

deg_120_n_fit_domain_start = 15
deg_120_n_fit_domain_end = 40

deg_120_n_fit_domain = n_vals[deg_120_n_fit_domain_start:deg_120_n_fit_domain_end]

deg_120_init_guesses = [65, 32, 2, 250, -14] #amp,mu,std dev,const,m_bg

deg_120_N_vals_fit = deg_120_N_vals[deg_120_n_fit_domain_start:deg_120_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_120_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_120_N_vals_fit)

fit_deg_120,cov_deg_120 = curve_fit(fit_gauss_lin_func, deg_120_n_fit_domain,
                                deg_120_N_vals_fit, deg_120_init_guesses)

deg_120_mu_err = np.sqrt(cov_deg_120[1,1])  #The uncertainty in the mean.
deg_120_sig_err = np.sqrt(cov_deg_120[2,2]) #The uncertainty in the std deviation.

deg_120_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_120_N_vals_fit,
                                                fit_gauss_lin_func(deg_120_n_fit_domain,
                                                                   fit_deg_120[0],
                                                                   fit_deg_120[1],
                                                                   fit_deg_120[2],
                                                                   fit_deg_120[3],
                                                                   fit_deg_120[4]),
                                                deg_120_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('120 Degrees Gaussian fit coefficients:')
print(fit_deg_120)
print('\n120 Degrees Covariance matrix:')
print(cov_deg_120)
print('\n120 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_120[1]} +/- {deg_120_mu_err}')
print('\n120 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_120[2]} +/- {deg_120_sig_err}')
print('\n120 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_120_rdd_chi_sqrd_val}')

deg_120_gauss_plot_domain = np.linspace(deg_120_n_fit_domain_start,
                                        deg_120_n_fit_domain_end-1,
                                        10_000)
deg_120_gauss_plot_vals = fit_gauss_lin_func(deg_120_gauss_plot_domain, amp=fit_deg_120[0],
                                        mu=fit_deg_120[1], sigma=fit_deg_120[2],
                                        const=fit_deg_120[3], m_bg=fit_deg_120[4])

plt.plot(deg_120_gauss_plot_domain, deg_120_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 130+10, 10))
plt.xlim(0,150)
#plt.ylim(0,130)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 120 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#130 degs:
plt.plot(n_vals,deg_130_N_vals, color='red', marker=".", linestyle='',
          label="130 degs")

deg_130_n_fit_domain_start = 25
deg_130_n_fit_domain_end = 35

deg_130_n_fit_domain = n_vals[deg_130_n_fit_domain_start:deg_130_n_fit_domain_end]

deg_130_init_guesses = [260, 30, 7, 300, -10] #amp,mu,std dev,const,m_bg

deg_130_N_vals_fit = deg_130_N_vals[deg_130_n_fit_domain_start:deg_130_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_130_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_130_N_vals_fit)

fit_deg_130,cov_deg_130 = curve_fit(fit_gauss_lin_func, deg_130_n_fit_domain,
                                deg_130_N_vals_fit, deg_130_init_guesses)

deg_130_mu_err = np.sqrt(cov_deg_130[1,1])  #The uncertainty in the mean.
deg_130_sig_err = np.sqrt(cov_deg_130[2,2]) #The uncertainty in the std deviation.

deg_130_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_130_N_vals_fit,
                                                fit_gauss_lin_func(deg_130_n_fit_domain,
                                                                   fit_deg_130[0],
                                                                   fit_deg_130[1],
                                                                   fit_deg_130[2],
                                                                   fit_deg_130[3],
                                                                   fit_deg_130[4]),
                                                deg_130_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('130 Degrees Gaussian fit coefficients:')
print(fit_deg_130)
print('\n130 Degrees Covariance matrix:')
print(cov_deg_130)
print('\n130 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_130[1]} +/- {deg_130_mu_err}')
print('\n130 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_130[2]} +/- {deg_130_sig_err}')
print('\n130 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_130_rdd_chi_sqrd_val}')

deg_130_gauss_plot_domain = np.linspace(deg_130_n_fit_domain_start,
                                        deg_130_n_fit_domain_end-1,
                                        10_000)
deg_130_gauss_plot_vals = fit_gauss_lin_func(deg_130_gauss_plot_domain, amp=fit_deg_130[0],
                                        mu=fit_deg_130[1], sigma=fit_deg_130[2],
                                        const=fit_deg_130[3], m_bg=fit_deg_130[4])

plt.plot(deg_130_gauss_plot_domain, deg_130_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 130+10, 10))
plt.xlim(0,150)
#plt.ylim(0,130)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 130 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#140 degs:
plt.plot(n_vals,deg_140_N_vals, color='red', marker=".", linestyle='',
          label="140 degs")

deg_140_n_fit_domain_start = 24
deg_140_n_fit_domain_end = 35

deg_140_n_fit_domain = n_vals[deg_140_n_fit_domain_start:deg_140_n_fit_domain_end]

deg_140_init_guesses = [350, 29, 7, 300, -10] #amp,mu,std dev,const,m_bg

deg_140_N_vals_fit = deg_140_N_vals[deg_140_n_fit_domain_start:deg_140_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_140_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_140_N_vals_fit)

fit_deg_140,cov_deg_140 = curve_fit(fit_gauss_lin_func, deg_140_n_fit_domain,
                                deg_140_N_vals_fit, deg_140_init_guesses)

deg_140_mu_err = np.sqrt(cov_deg_140[1,1])  #The uncertainty in the mean.
deg_140_sig_err = np.sqrt(cov_deg_140[2,2]) #The uncertainty in the std deviation.

deg_140_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_140_N_vals_fit,
                                                fit_gauss_lin_func(deg_140_n_fit_domain,
                                                                   fit_deg_140[0],
                                                                   fit_deg_140[1],
                                                                   fit_deg_140[2],
                                                                   fit_deg_140[3],
                                                                   fit_deg_140[4]),
                                                deg_140_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('140 Degrees Gaussian fit coefficients:')
print(fit_deg_140)
print('\n140 Degrees Covariance matrix:')
print(cov_deg_140)
print('\n140 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_140[1]} +/- {deg_140_mu_err}')
print('\n140 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_140[2]} +/- {deg_140_sig_err}')
print('\n140 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_140_rdd_chi_sqrd_val}')

deg_140_gauss_plot_domain = np.linspace(deg_140_n_fit_domain_start,
                                        deg_140_n_fit_domain_end-1,
                                        10_000)
deg_140_gauss_plot_vals = fit_gauss_lin_func(deg_140_gauss_plot_domain, amp=fit_deg_140[0],
                                        mu=fit_deg_140[1], sigma=fit_deg_140[2],
                                        const=fit_deg_140[3], m_bg=fit_deg_140[4])

plt.plot(deg_140_gauss_plot_domain, deg_140_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 150+10, 10))
plt.xlim(0,150)
#plt.ylim(0,130)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 140 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#150 degs:
plt.plot(n_vals,deg_150_N_vals, color='red', marker=".", linestyle='',
          label="150 degs")

deg_150_n_fit_domain_start = 23
deg_150_n_fit_domain_end = 33

deg_150_n_fit_domain = n_vals[deg_150_n_fit_domain_start:deg_150_n_fit_domain_end]

deg_150_init_guesses = [400, 28, 7, 300, -8] #amp,mu,std dev,const,m_bg

deg_150_N_vals_fit = deg_150_N_vals[deg_150_n_fit_domain_start:deg_150_n_fit_domain_end]

#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
deg_150_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(deg_150_N_vals_fit)

fit_deg_150,cov_deg_150 = curve_fit(fit_gauss_lin_func, deg_150_n_fit_domain,
                                deg_150_N_vals_fit, deg_150_init_guesses)

deg_150_mu_err = np.sqrt(cov_deg_150[1,1])  #The uncertainty in the mean.
deg_150_sig_err = np.sqrt(cov_deg_150[2,2]) #The uncertainty in the std deviation.

deg_150_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(deg_150_N_vals_fit,
                                                fit_gauss_lin_func(deg_150_n_fit_domain,
                                                                   fit_deg_150[0],
                                                                   fit_deg_150[1],
                                                                   fit_deg_150[2],
                                                                   fit_deg_150[3],
                                                                   fit_deg_150[4]),
                                                deg_150_sqr_rt_N_vals_fit,5)

print('----------------------------------------------------------------------')
print('150 Degrees Gaussian fit coefficients:')
print(fit_deg_150)
print('\n150 Degrees Covariance matrix:')
print(cov_deg_150)
print('\n150 Degrees Mean Bin Number and Error:')
print(f'mean n of final photon energy = {fit_deg_150[1]} +/- {deg_150_mu_err}')
print('\n150 Degrees Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_deg_150[2]} +/- {deg_150_sig_err}')
print('\n150 Degrees Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {deg_150_rdd_chi_sqrd_val}')

deg_150_gauss_plot_domain = np.linspace(deg_150_n_fit_domain_start,
                                        deg_150_n_fit_domain_end-1,
                                        10_000)
deg_150_gauss_plot_vals = fit_gauss_lin_func(deg_150_gauss_plot_domain, amp=fit_deg_150[0],
                                        mu=fit_deg_150[1], sigma=fit_deg_150[2],
                                        const=fit_deg_150[3], m_bg=fit_deg_150[4])

plt.plot(deg_150_gauss_plot_domain, deg_150_gauss_plot_vals, 'black', alpha=0.8,
          linestyle='dashed', label='Gaussian Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 150+10, 10))
#plt.yticks(np.arange(0, 130+10, 10))
plt.xlim(0,150)
#plt.ylim(0,130)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V 150 Degrees Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#%%----------------------------------------------------------------------------
#-------------------------------Calculations:----------------------------------
def get_pk_n_errs(sigma_vals_array):
    "RETURNS: Array of errors for peak energy n-values."
    pk_n_errs_array = (0.5)*2*np.sqrt(2*np.log(2)) * sigma_vals_array
    return pk_n_errs_array

def get_energy_units(n_array):
    "RETURNS: Array of values of unit 'n' to keV."
    rtrn_energy_list = []
    n_list = list(n_array)
    for i in range(0,len(n_list)):
        energy_val = (7.244520759572183*n_list[i]) + -9.175764370840774
        rtrn_energy_list.append(energy_val)
    rtrn_energy_array = np.array(rtrn_energy_list)
    return rtrn_energy_array

def get_energy_errs_units(n_array):
    "RETURNS: Array of values of unit 'n' to keV."
    rtrn_energy_list = []
    n_list = list(n_array)
    for i in range(0,len(n_list)):
        energy_val = (7.244520759572183*n_list[i])
        rtrn_energy_list.append(energy_val)
    rtrn_energy_array = np.array(rtrn_energy_list)
    return rtrn_energy_array

#Putting all sigmas into ordered array:
sigma_vals = [fit_deg_10[2],fit_deg_20[2],fit_deg_30[2],fit_deg_40[2],
              fit_deg_50[2],fit_deg_60[2],fit_deg_70[2],
              fit_deg_80[2],fit_deg_90[2], fit_deg_100[2],fit_deg_110[2],
              fit_deg_120[2],fit_deg_130[2],fit_deg_140[2],fit_deg_150[2]]

sigma_vals = np.array(sigma_vals)
sigma_vals = np.abs(sigma_vals)
pk_n_errs = get_pk_n_errs(sigma_vals)
#------------------------------------------------------------------------------
#Putting all final photon peak n-values into ordered array:
pk_n_vals = [fit_deg_10[1],fit_deg_20[1],fit_deg_30[1],fit_deg_40[1],
              fit_deg_50[1],fit_deg_60[1],fit_deg_70[1],
              fit_deg_80[1],fit_deg_90[1],fit_deg_100[1],fit_deg_110[1],
              fit_deg_120[1],fit_deg_130[1],fit_deg_140[1],fit_deg_150[1]]

fnl_phtn_energs = get_energy_units(pk_n_vals)
fnl_phtn_energs_errs = get_energy_errs_units(pk_n_errs)
#------------------------------------------------------------------------------
#Gaining angle array and errors:
angles = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150] #In degrees
angles = np.array(angles)

angle_errs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] #In degrees
angle_errs = np.array(angle_errs)
#%%----------------------------------------------------------------------------
#-------------------------------Plotting:--------------------------------------
plt.errorbar(angles, fnl_phtn_energs, color='red', marker=".",
         linestyle='', label="Experiment Data", yerr=fnl_phtn_energs_errs,
         capsize=4, xerr=angle_errs, ecolor='black')

#Plotting Fit:
def get_theory_func(E_gam0_val, elec_rest_energy_val, fit_angle_domain_array):
    "RETURNS: Array of theoretical final fitted photon energy values."
    rad_angle_domain_array = (fit_angle_domain_array/180) * np.pi
    rtrn_theory_energy_list = []
    rad_angle_domain_list = list(rad_angle_domain_array)
    for i in range(0,len(rad_angle_domain_list)):
        num = E_gam0_val
        denom = 1 + ((E_gam0_val/elec_rest_energy_val)*(1-np.cos(rad_angle_domain_list[i])))
        theory_energy_val = num/denom
        rtrn_theory_energy_list.append(theory_energy_val)
    rtrn_theory_energy_array = np.array(rtrn_theory_energy_list)
    return rtrn_theory_energy_array

fit_angle_domain = np.linspace(0, 160, 10_000)
E_gam0 = 661.7 #keV
elec_rest_energy = 511 #keV

fit_final_photon_energy_vals = get_theory_func(E_gam0, elec_rest_energy,
                                               fit_angle_domain)

plt.plot(fit_angle_domain, fit_final_photon_energy_vals, 'blue', alpha=0.8,
         linestyle='dashed', label='Theoretical Fit')

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
plt.xticks(np.arange(0, 160+10, 10))
plt.yticks(np.arange(150, 700+25, 25))
plt.xlim(0,160)
plt.ylim(150,700)
plt.xlabel("Scattering Angle," " " r"${\theta}$" " " "(deg)")
plt.ylabel("Final Photon Energy," " " r"$E_{\gamma}$" r"$({\theta})$" " " "(keV)")
plt.grid()
plt.legend()
#plt.savefig("Set 1 Scattering Angle Plot",dpi=600, bbox_inches="tight")
plt.show()
#%%----------------------------------------------------------------------------
#-------------------------Compton Goodness of Fit:-----------------------------
compton_theory_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(fnl_phtn_energs,
                                                       get_theory_func(E_gam0,
                                                                       elec_rest_energy,
                                                                       angles), 
                                                       fnl_phtn_energs_errs, num_free_params=0)

print('\nReduced Chi-squared of Compton Theory Fit:')
print(f'Reduced Chi-squared = {compton_theory_rdd_chi_sqrd_val}')
print('\n--------------------------------------------------------------------')
#%%----------------------------------------------------------------------------
#-----------------------Differential Cross-section Code:-----------------------
def get_lin_area(n_domain, const, grad):
    '''Returns value of area under linear line.'''
    linear_values = grad*n_domain + const
    area = trapz(linear_values, n_domain)
    return area

#Import Time Measurement Data:
deg_10_meas_tm = 200
deg_20_meas_tm = 200
deg_30_meas_tm = 200
deg_40_meas_tm = 200
deg_50_meas_tm = 300
deg_60_meas_tm = 300
deg_70_meas_tm = 300      #All in seconds
deg_80_meas_tm = 300
deg_90_meas_tm = 400
deg_100_meas_tm = 600
deg_110_meas_tm = 600
deg_120_meas_tm = 600
deg_130_meas_tm = 600
deg_140_meas_tm = 600
deg_150_meas_tm = 600

#Total Scattering Event Counts (Area of Gaussians):
deg_10_gauss_area = trapz(deg_10_gauss_plot_vals,deg_10_gauss_plot_domain)
deg_10_lin_area = get_lin_area(deg_10_gauss_plot_domain, fit_deg_10[3], fit_deg_10[4])
deg_10_gauss_area -= deg_10_lin_area
deg_10_gauss_area = deg_10_gauss_area*0.76

deg_20_gauss_area = trapz(deg_20_gauss_plot_vals,deg_20_gauss_plot_domain)
deg_20_lin_area = get_lin_area(deg_20_gauss_plot_domain, fit_deg_20[3], fit_deg_20[4])
deg_20_gauss_area -= deg_20_lin_area
deg_20_gauss_area = deg_20_gauss_area*0.76

deg_30_gauss_area = trapz(deg_30_gauss_plot_vals,deg_30_gauss_plot_domain)
deg_30_lin_area = get_lin_area(deg_30_gauss_plot_domain, fit_deg_30[3], fit_deg_30[4])
deg_30_gauss_area -= deg_30_lin_area
deg_30_gauss_area = deg_30_gauss_area*0.76

deg_40_gauss_area = trapz(deg_40_gauss_plot_vals,deg_40_gauss_plot_domain)
deg_40_lin_area = get_lin_area(deg_40_gauss_plot_domain, fit_deg_40[3], fit_deg_40[4])
deg_40_gauss_area -= deg_40_lin_area
deg_40_gauss_area = deg_40_gauss_area*0.76

deg_50_gauss_area = trapz(deg_50_gauss_plot_vals,deg_50_gauss_plot_domain)
deg_50_lin_area = get_lin_area(deg_50_gauss_plot_domain, fit_deg_50[3], fit_deg_50[4])
deg_50_gauss_area -= deg_50_lin_area
deg_50_gauss_area = deg_50_gauss_area*0.76

deg_60_gauss_area = trapz(deg_60_gauss_plot_vals,deg_60_gauss_plot_domain)
deg_60_lin_area = get_lin_area(deg_60_gauss_plot_domain, fit_deg_60[3], fit_deg_60[4])
deg_60_gauss_area -= deg_60_lin_area
deg_60_gauss_area = deg_60_gauss_area*0.76

deg_70_gauss_area = trapz(deg_70_gauss_plot_vals,deg_70_gauss_plot_domain)
deg_70_lin_area = get_lin_area(deg_70_gauss_plot_domain, fit_deg_70[3], fit_deg_70[4])
deg_70_gauss_area -= deg_70_lin_area
deg_70_gauss_area = deg_70_gauss_area*0.76

deg_80_gauss_area = trapz(deg_80_gauss_plot_vals,deg_80_gauss_plot_domain)
deg_80_lin_area = get_lin_area(deg_80_gauss_plot_domain, fit_deg_80[3], fit_deg_80[4])
deg_80_gauss_area -= deg_80_lin_area
deg_80_big_gauss_area = deg_80_gauss_area*0.76

deg_90_gauss_area = trapz(deg_90_gauss_plot_vals,deg_90_gauss_plot_domain)
deg_90_lin_area = get_lin_area(deg_90_gauss_plot_domain, fit_deg_90[3], fit_deg_90[4])
deg_90_gauss_area -= deg_90_lin_area
deg_90_gauss_area = deg_90_gauss_area*0.76

deg_100_gauss_area = trapz(deg_100_gauss_plot_vals,deg_100_gauss_plot_domain)
deg_100_lin_area = get_lin_area(deg_100_gauss_plot_domain, fit_deg_100[3], fit_deg_100[4])
deg_100_gauss_area -= deg_100_lin_area
deg_100_gauss_area = deg_100_gauss_area*0.76

deg_110_gauss_area = trapz(deg_110_gauss_plot_vals,deg_110_gauss_plot_domain)
deg_110_lin_area = get_lin_area(deg_110_gauss_plot_domain, fit_deg_110[3], fit_deg_110[4])
deg_110_gauss_area -= deg_110_lin_area
deg_110_gauss_area = deg_110_gauss_area*0.76

deg_120_gauss_area = trapz(deg_120_gauss_plot_vals,deg_120_gauss_plot_domain)
deg_120_lin_area = get_lin_area(deg_120_gauss_plot_domain, fit_deg_120[3], fit_deg_120[4])
deg_120_gauss_area -= deg_120_lin_area
deg_120_gauss_area = deg_120_gauss_area*0.76

deg_130_gauss_area = trapz(deg_130_gauss_plot_vals,deg_130_gauss_plot_domain)
deg_130_lin_area = get_lin_area(deg_130_gauss_plot_domain, fit_deg_130[3], fit_deg_130[4])
deg_130_gauss_area -= deg_130_lin_area
deg_130_gauss_area = deg_130_gauss_area*0.76

deg_140_gauss_area = trapz(deg_140_gauss_plot_vals,deg_140_gauss_plot_domain)
deg_140_lin_area = get_lin_area(deg_140_gauss_plot_domain, fit_deg_140[3], fit_deg_140[4])
deg_140_gauss_area -= deg_140_lin_area
deg_140_gauss_area = deg_140_gauss_area*0.76

deg_150_gauss_area = trapz(deg_150_gauss_plot_vals,deg_150_gauss_plot_domain)
deg_150_lin_area = get_lin_area(deg_150_gauss_plot_domain, fit_deg_150[3], fit_deg_150[4])
deg_150_gauss_area -= deg_150_lin_area
deg_150_gauss_area = deg_150_gauss_area*0.76

#Rate of Scattering Events:
deg_10_sctr_rate = deg_10_gauss_area / deg_10_meas_tm
deg_20_sctr_rate = deg_20_gauss_area / deg_20_meas_tm
deg_30_sctr_rate = deg_30_gauss_area / deg_30_meas_tm
deg_40_sctr_rate = deg_40_gauss_area / deg_40_meas_tm
deg_50_sctr_rate = deg_50_gauss_area / deg_50_meas_tm
deg_60_sctr_rate = deg_60_gauss_area / deg_60_meas_tm       #N_tot_sctr/s
deg_70_sctr_rate = deg_70_gauss_area / deg_70_meas_tm
deg_80_sctr_rate = deg_80_gauss_area / deg_80_meas_tm
deg_90_sctr_rate_org = deg_90_gauss_area / deg_90_meas_tm #Want to keep this value for later.
deg_100_sctr_rate = deg_100_gauss_area / deg_100_meas_tm
deg_110_sctr_rate = deg_110_gauss_area / deg_110_meas_tm
deg_120_sctr_rate = deg_120_gauss_area / deg_120_meas_tm
deg_130_sctr_rate = deg_130_gauss_area / deg_130_meas_tm
deg_140_sctr_rate = deg_140_gauss_area / deg_140_meas_tm
deg_150_sctr_rate = deg_150_gauss_area / deg_150_meas_tm
#%%----------------------------------------------------------------------------
#------------------------------Calculations:-----------------------------------
def gain_norm_diff_cross_area_val(deg_sctr_rate, denom_deg_sctr_rate):
    "RETURNS: Normalised Differential Cross-section Value."
    deg_norm_diff_cross_area_val = deg_sctr_rate / denom_deg_sctr_rate
    return deg_norm_diff_cross_area_val

deg_10_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_10_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_20_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_20_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_30_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_30_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_40_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_40_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_50_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_50_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_60_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_60_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_70_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_70_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_80_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_80_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_90_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_90_sctr_rate_org,
                                                            deg_90_sctr_rate_org)

deg_100_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_100_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_110_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_110_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_120_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_120_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_130_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_130_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_140_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_140_sctr_rate,
                                                            deg_90_sctr_rate_org)

deg_150_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_150_sctr_rate,
                                                            deg_90_sctr_rate_org)

#Forming an array of the normalised differential cross-sections:
norm_diff_cross_areas = [deg_10_norm_diff_cross_area,
                         deg_20_norm_diff_cross_area,
                         deg_30_norm_diff_cross_area,
                         deg_40_norm_diff_cross_area,
                         deg_50_norm_diff_cross_area,
                         deg_60_norm_diff_cross_area,
                         deg_70_norm_diff_cross_area,
                         deg_80_norm_diff_cross_area,
                         #deg_90_norm_diff_cross_area,
                         deg_100_norm_diff_cross_area,
                         deg_110_norm_diff_cross_area,
                         deg_120_norm_diff_cross_area,
                         deg_130_norm_diff_cross_area,
                         deg_140_norm_diff_cross_area,
                         deg_150_norm_diff_cross_area]

norm_diff_cross_areas = np.array(norm_diff_cross_areas)

norm_diff_cross_areas_angles = [10,20,30,40,50,60,70,80,100,110,120,130,140,150]
#In degrees
#%%----------------------------------------------------------------------------
#--------------------------------Errors:---------------------------------------
norm_diff_cross_areas_angles_errs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2]

#Gain areas under FWHM of biggest gaussians possible:
#(Use largest amplitude and std deviation to form gaussians).
deg_10_big_gauss_plot_vals = fit_gauss_lin_func(deg_10_gauss_plot_domain,
                                            amp=fit_deg_10[0]+np.sqrt(cov_deg_10[0,0]),
                                            mu=fit_deg_10[1],
                                            sigma=fit_deg_10[2]+deg_10_sig_err,
                                            const=fit_deg_10[3], m_bg=fit_deg_10[4])

deg_20_big_gauss_plot_vals = fit_gauss_lin_func(deg_20_gauss_plot_domain,
                                            amp=fit_deg_20[0]+np.sqrt(cov_deg_20[0,0]),
                                            mu=fit_deg_20[1],
                                            sigma=fit_deg_20[2]+deg_20_sig_err,
                                            const=fit_deg_20[3], m_bg=fit_deg_20[4])

deg_30_big_gauss_plot_vals = fit_gauss_lin_func(deg_30_gauss_plot_domain,
                                            amp=fit_deg_30[0]+np.sqrt(cov_deg_30[0,0]),
                                            mu=fit_deg_30[1],
                                            sigma=fit_deg_30[2]+deg_30_sig_err,
                                            const=fit_deg_30[3], m_bg=fit_deg_30[4])

deg_40_big_gauss_plot_vals = fit_gauss_lin_func(deg_40_gauss_plot_domain,
                                            amp=fit_deg_40[0]+np.sqrt(cov_deg_40[0,0]),
                                            mu=fit_deg_40[1],
                                            sigma=fit_deg_40[2]+deg_40_sig_err,
                                            const=fit_deg_40[3], m_bg=fit_deg_40[4])

deg_50_big_gauss_plot_vals = fit_gauss_lin_func(deg_50_gauss_plot_domain,
                                            amp=fit_deg_50[0]+np.sqrt(cov_deg_50[0,0]),
                                            mu=fit_deg_50[1],
                                            sigma=fit_deg_50[2]+deg_50_sig_err,
                                            const=fit_deg_50[3], m_bg=fit_deg_50[4])

deg_60_big_gauss_plot_vals = fit_gauss_lin_func(deg_60_gauss_plot_domain,
                                            amp=fit_deg_60[0]+np.sqrt(cov_deg_60[0,0]),
                                            mu=fit_deg_60[1],
                                            sigma=fit_deg_60[2]+deg_60_sig_err,
                                            const=fit_deg_60[3], m_bg=fit_deg_60[4])

deg_70_big_gauss_plot_vals = fit_gauss_lin_func(deg_70_gauss_plot_domain,
                                            amp=fit_deg_70[0]+np.sqrt(cov_deg_70[0,0]),
                                            mu=fit_deg_70[1],
                                            sigma=fit_deg_70[2]+deg_70_sig_err,
                                            const=fit_deg_70[3], m_bg=fit_deg_70[4])

deg_80_big_gauss_plot_vals = fit_gauss_lin_func(deg_80_gauss_plot_domain,
                                            amp=fit_deg_80[0]+np.sqrt(cov_deg_80[0,0]),
                                            mu=fit_deg_80[1],
                                            sigma=fit_deg_80[2]+deg_80_sig_err,
                                            const=fit_deg_80[3], m_bg=fit_deg_80[4])

deg_90_big_gauss_plot_vals = fit_gauss_lin_func(deg_90_gauss_plot_domain,
                                            amp=fit_deg_90[0]+np.sqrt(cov_deg_90[0,0]),
                                            mu=fit_deg_90[1],
                                            sigma=fit_deg_90[2]+deg_90_sig_err,
                                            const=fit_deg_90[3], m_bg=fit_deg_90[4])

deg_100_big_gauss_plot_vals = fit_gauss_lin_func(deg_100_gauss_plot_domain,
                                            amp=fit_deg_100[0]+np.sqrt(cov_deg_100[0,0]),
                                            mu=fit_deg_100[1],
                                            sigma=fit_deg_100[2]+deg_100_sig_err,
                                            const=fit_deg_100[3], m_bg=fit_deg_100[4])

deg_110_big_gauss_plot_vals = fit_gauss_lin_func(deg_110_gauss_plot_domain,
                                            amp=fit_deg_110[0]+np.sqrt(cov_deg_110[0,0]),
                                            mu=fit_deg_110[1],
                                            sigma=fit_deg_110[2]+deg_110_sig_err,
                                            const=fit_deg_110[3], m_bg=fit_deg_110[4])

deg_120_big_gauss_plot_vals = fit_gauss_lin_func(deg_120_gauss_plot_domain,
                                            amp=fit_deg_120[0]+np.sqrt(cov_deg_120[0,0]),
                                            mu=fit_deg_120[1],
                                            sigma=fit_deg_120[2]+deg_120_sig_err,
                                            const=fit_deg_120[3], m_bg=fit_deg_120[4])

deg_130_big_gauss_plot_vals = fit_gauss_lin_func(deg_130_gauss_plot_domain,
                                            amp=fit_deg_130[0]+np.sqrt(cov_deg_130[0,0]),
                                            mu=fit_deg_130[1],
                                            sigma=fit_deg_130[2]+deg_130_sig_err,
                                            const=fit_deg_130[3], m_bg=fit_deg_130[4])

deg_140_big_gauss_plot_vals = fit_gauss_lin_func(deg_140_gauss_plot_domain,
                                            amp=fit_deg_140[0]+np.sqrt(cov_deg_140[0,0]),
                                            mu=fit_deg_140[1],
                                            sigma=fit_deg_140[2]+deg_140_sig_err,
                                            const=fit_deg_140[3], m_bg=fit_deg_140[4])

deg_150_big_gauss_plot_vals = fit_gauss_lin_func(deg_150_gauss_plot_domain,
                                            amp=fit_deg_150[0]+np.sqrt(cov_deg_150[0,0]),
                                            mu=fit_deg_150[1],
                                            sigma=fit_deg_150[2]+deg_150_sig_err,
                                            const=fit_deg_150[3], m_bg=fit_deg_150[4])
#------------------------------------------------------------------------------
#Have to gain area under the FWHM of BIG gaussians:
#Total Scattering Event Counts (Area of Gaussians):
#------------------------------------------------------------------------------
#Gaining under FWHM of BIG gaussians:
deg_10_big_gauss_area = trapz(deg_10_big_gauss_plot_vals,deg_10_gauss_plot_domain)
deg_10_lin_area = get_lin_area(deg_10_gauss_plot_domain, fit_deg_10[3], fit_deg_10[4])
deg_10_big_gauss_area -= deg_10_lin_area
deg_10_big_gauss_area = deg_10_big_gauss_area*0.76

deg_20_big_gauss_area = trapz(deg_20_big_gauss_plot_vals,deg_20_gauss_plot_domain)
deg_20_lin_area = get_lin_area(deg_20_gauss_plot_domain, fit_deg_20[3], fit_deg_20[4])
deg_20_big_gauss_area -= deg_20_lin_area
deg_20_big_gauss_area = deg_20_big_gauss_area*0.76

deg_30_big_gauss_area = trapz(deg_30_big_gauss_plot_vals,deg_30_gauss_plot_domain)
deg_30_lin_area = get_lin_area(deg_30_gauss_plot_domain, fit_deg_30[3], fit_deg_30[4])
deg_30_big_gauss_area -= deg_30_lin_area
deg_30_big_gauss_area = deg_30_big_gauss_area*0.76

deg_40_big_gauss_area = trapz(deg_40_big_gauss_plot_vals,deg_40_gauss_plot_domain)
deg_40_lin_area = get_lin_area(deg_40_gauss_plot_domain, fit_deg_40[3], fit_deg_40[4])
deg_40_big_gauss_area -= deg_40_lin_area
deg_40_big_gauss_area = deg_40_big_gauss_area*0.76

deg_50_big_gauss_area = trapz(deg_50_big_gauss_plot_vals,deg_50_gauss_plot_domain)
deg_50_lin_area = get_lin_area(deg_50_gauss_plot_domain, fit_deg_50[3], fit_deg_50[4])
deg_50_big_gauss_area -= deg_50_lin_area
deg_50_big_gauss_area = deg_50_big_gauss_area*0.76

deg_60_big_gauss_area = trapz(deg_60_big_gauss_plot_vals,deg_60_gauss_plot_domain)
deg_60_lin_area = get_lin_area(deg_60_gauss_plot_domain, fit_deg_60[3], fit_deg_60[4])
deg_60_big_gauss_area -= deg_60_lin_area
deg_60_big_gauss_area = deg_60_big_gauss_area*0.76

deg_70_big_gauss_area = trapz(deg_70_big_gauss_plot_vals,deg_70_gauss_plot_domain)
deg_70_lin_area = get_lin_area(deg_70_gauss_plot_domain, fit_deg_70[3], fit_deg_70[4])
deg_70_big_gauss_area -= deg_70_lin_area
deg_70_big_gauss_area = deg_70_big_gauss_area*0.76

deg_80_big_gauss_area = trapz(deg_80_big_gauss_plot_vals,deg_80_gauss_plot_domain)
deg_80_lin_area = get_lin_area(deg_80_gauss_plot_domain, fit_deg_80[3], fit_deg_80[4])
deg_80_big_gauss_area -= deg_80_lin_area
deg_80_big_gauss_area = deg_80_big_gauss_area*0.76

deg_90_big_gauss_area = trapz(deg_90_big_gauss_plot_vals,deg_90_gauss_plot_domain)
deg_90_lin_area = get_lin_area(deg_90_gauss_plot_domain, fit_deg_90[3], fit_deg_90[4])
deg_90_big_gauss_area -= deg_90_lin_area
deg_90_big_gauss_area = deg_90_big_gauss_area*0.76

deg_100_big_gauss_area = trapz(deg_100_big_gauss_plot_vals,deg_100_gauss_plot_domain)
deg_100_lin_area = get_lin_area(deg_100_gauss_plot_domain, fit_deg_100[3], fit_deg_100[4])
deg_100_big_gauss_area -= deg_100_lin_area
deg_100_big_gauss_area = deg_100_big_gauss_area*0.76

deg_110_big_gauss_area = trapz(deg_110_big_gauss_plot_vals,deg_110_gauss_plot_domain)
deg_110_lin_area = get_lin_area(deg_110_gauss_plot_domain, fit_deg_110[3], fit_deg_110[4])
deg_110_big_gauss_area -= deg_110_lin_area
deg_110_big_gauss_area = deg_110_big_gauss_area*0.76

deg_120_big_gauss_area = trapz(deg_120_big_gauss_plot_vals,deg_120_gauss_plot_domain)
deg_120_lin_area = get_lin_area(deg_120_gauss_plot_domain, fit_deg_120[3], fit_deg_120[4])
deg_120_big_gauss_area -= deg_120_lin_area
deg_120_big_gauss_area = deg_120_big_gauss_area*0.76

deg_130_big_gauss_area = trapz(deg_130_big_gauss_plot_vals,deg_130_gauss_plot_domain)
deg_130_lin_area = get_lin_area(deg_130_gauss_plot_domain, fit_deg_130[3], fit_deg_130[4])
deg_130_big_gauss_area -= deg_130_lin_area
deg_130_big_gauss_area = deg_130_big_gauss_area*0.76

deg_140_big_gauss_area = trapz(deg_140_big_gauss_plot_vals,deg_140_gauss_plot_domain)
deg_140_lin_area = get_lin_area(deg_140_gauss_plot_domain, fit_deg_140[3], fit_deg_140[4])
deg_140_big_gauss_area -= deg_140_lin_area
deg_140_big_gauss_area = deg_140_big_gauss_area*0.76

deg_150_big_gauss_area = trapz(deg_150_big_gauss_plot_vals,deg_150_gauss_plot_domain)
deg_150_lin_area = get_lin_area(deg_150_gauss_plot_domain, fit_deg_150[3], fit_deg_150[4])
deg_150_big_gauss_area -= deg_150_lin_area
deg_150_big_gauss_area = deg_150_big_gauss_area*0.76
#------------------------------------------------------------------------------
#Rate of Scattering Events:
deg_10_sctr_rate = deg_10_big_gauss_area / deg_10_meas_tm
deg_20_sctr_rate = deg_20_big_gauss_area / deg_20_meas_tm                   
deg_30_sctr_rate = deg_30_big_gauss_area / deg_30_meas_tm
deg_40_sctr_rate = deg_40_big_gauss_area / deg_40_meas_tm
deg_50_sctr_rate = deg_50_big_gauss_area / deg_50_meas_tm
deg_60_sctr_rate = deg_60_big_gauss_area / deg_60_meas_tm       #N_tot_sctr/s
deg_70_sctr_rate = deg_70_big_gauss_area / deg_70_meas_tm
deg_80_sctr_rate = deg_80_big_gauss_area / deg_80_meas_tm
deg_90_sctr_rate = deg_90_sctr_rate_org                         #Want to use original.
deg_100_sctr_rate = deg_100_big_gauss_area / deg_100_meas_tm
deg_110_sctr_rate = deg_110_big_gauss_area / deg_110_meas_tm
deg_120_sctr_rate = deg_120_big_gauss_area / deg_120_meas_tm
deg_130_sctr_rate = deg_130_big_gauss_area / deg_130_meas_tm
deg_140_sctr_rate = deg_140_big_gauss_area / deg_140_meas_tm
deg_150_sctr_rate = deg_150_big_gauss_area / deg_150_meas_tm

deg_10_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_10_sctr_rate,
                                                            deg_90_sctr_rate)

deg_20_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_20_sctr_rate,
                                                            deg_90_sctr_rate)

deg_30_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_30_sctr_rate,
                                                            deg_90_sctr_rate)

deg_40_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_40_sctr_rate,
                                                            deg_90_sctr_rate)

deg_50_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_50_sctr_rate,
                                                            deg_90_sctr_rate)

deg_60_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_60_sctr_rate,
                                                            deg_90_sctr_rate)

deg_70_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_70_sctr_rate,
                                                            deg_90_sctr_rate)

deg_80_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_80_sctr_rate,
                                                            deg_90_sctr_rate)

deg_90_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_90_sctr_rate,
                                                            deg_90_sctr_rate)

deg_100_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_100_sctr_rate,
                                                            deg_90_sctr_rate)

deg_110_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_110_sctr_rate,
                                                            deg_90_sctr_rate)

deg_120_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_120_sctr_rate,
                                                            deg_90_sctr_rate)

deg_130_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_130_sctr_rate,
                                                            deg_90_sctr_rate)

deg_140_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_140_sctr_rate,
                                                            deg_90_sctr_rate)

deg_150_norm_diff_cross_area = gain_norm_diff_cross_area_val(deg_150_sctr_rate,
                                                            deg_90_sctr_rate)

#Forming an array of the normalised differential cross-sections:
big_gauss_norm_diff_cross_areas = [deg_10_norm_diff_cross_area,
                                   deg_20_norm_diff_cross_area,
                                   deg_30_norm_diff_cross_area,
                                   deg_40_norm_diff_cross_area,
                                   deg_50_norm_diff_cross_area,
                                   deg_60_norm_diff_cross_area,
                                   deg_70_norm_diff_cross_area,
                                   deg_80_norm_diff_cross_area,
                                   #deg_90_norm_diff_cross_area,
                                   deg_100_norm_diff_cross_area,
                                   deg_110_norm_diff_cross_area,
                                   deg_120_norm_diff_cross_area,
                                   deg_130_norm_diff_cross_area,
                                   deg_140_norm_diff_cross_area,
                                   deg_150_norm_diff_cross_area]

big_gauss_norm_diff_cross_areas = np.array(big_gauss_norm_diff_cross_areas)

norm_diff_cross_areas_errs = np.abs(big_gauss_norm_diff_cross_areas - norm_diff_cross_areas)
#%%----------------------------------------------------------------------------
#-------------------------------Plotting:--------------------------------------
#Plotting Data:
#Not using 10 and 20 degrees data.
#Use data points that we want to use:
plt.errorbar(norm_diff_cross_areas_angles[2:],norm_diff_cross_areas[2:],
             color='red', marker=".", linestyle='', label="Experiment Data",
             yerr=norm_diff_cross_areas_errs[2:], capsize=4,
             xerr=norm_diff_cross_areas_angles_errs[2:], ecolor='black')

#Plotting Fits:
def get_thomson_fit_func(fit_angle_domain_array, norm_deg_val):
    "RETURNS: Array of Thomson Normalised Differential Cross-section values."
    return_thomson_fit_list = []
    rad_fit_angle_domain_list = []
    for i in range(0,len(list(fit_angle_domain_array))):
        rad_fit_angle_domain_val = list(fit_angle_domain_array)[i]*np.pi/180
        rad_fit_angle_domain_list.append(rad_fit_angle_domain_val)
    rad_norm_deg_val = norm_deg_val*np.pi/180
    for i in range(0,len(rad_fit_angle_domain_list)):
        num = 1 + np.cos(rad_fit_angle_domain_list[i])**2
        denom = 1 + np.cos(rad_norm_deg_val)**2
        thomson_fit_val = num/denom
        return_thomson_fit_list.append(thomson_fit_val)
    return_thomson_fit_array = np.array(return_thomson_fit_list)
    return return_thomson_fit_array

def get_f_func_val(fit_angle_val, E_gam_val):
    "RETURNS: VALUE of f-function for Klein-Nishina Equation."
    num = 1
    denom = 1 + (E_gam_val*(1-np.cos(fit_angle_val)))
    f_func_val = num/denom
    return f_func_val

def get_klein_fit_func(fit_angle_domain_array, norm_deg_val, E_gam_val):
    "RETURNS: Array of Klein Normalised Differential Cross-section values."
    return_klein_fit_list = []
    rad_fit_angle_domain_list = []
    for i in range(0,len(list(fit_angle_domain_array))):
        rad_fit_angle_domain_val = list(fit_angle_domain_array)[i]*np.pi/180
        rad_fit_angle_domain_list.append(rad_fit_angle_domain_val)
    rad_norm_deg_val = norm_deg_val*np.pi/180
    for i in range(0,len(rad_fit_angle_domain_list)):
        f_func_val = get_f_func_val(rad_fit_angle_domain_list[i], E_gam_val)
        num = (f_func_val**2)*(f_func_val + (1/f_func_val) - (np.sin(rad_fit_angle_domain_list[i])**2))
        f_func_val = get_f_func_val(rad_norm_deg_val, E_gam_val)
        denom = (f_func_val**2)*(f_func_val + (1/f_func_val) - (np.sin(rad_norm_deg_val**2)))
        klein_fit_val = num/denom
        return_klein_fit_list.append(klein_fit_val)
    return_klein_fit_array = np.array(return_klein_fit_list)
    return return_klein_fit_array
    
fit_angle_domain = np.linspace(20, 160, 10_000)

fit_thomson_vals = get_thomson_fit_func(fit_angle_domain, 90)
plt.plot(fit_angle_domain, fit_thomson_vals, 'orange', alpha=0.8,
         linestyle='dashed', label='Thomson Theory Fit')

#For Klein-Nishina Theory Formula:
E_gam = 661.7/511 
fit_klein_vals = get_klein_fit_func(fit_angle_domain, 90, E_gam)
plt.plot(fit_angle_domain, fit_klein_vals, 'blue', alpha=0.8,
         linestyle='dashed', label='Klein-Nishina Theory Fit')

plt.xticks(np.arange(20, 150+10, 10))
plt.yticks(np.arange(0, 2.5+0.25, 0.25))
plt.xlim(20,160)
plt.ylim(0,2.5)
plt.xlabel("Scattering Angle," " " r"${\theta}$" " " "(deg)", fontsize=MEDIUM_SIZE)
plt.ylabel("Normalised Differential Cross-section", fontsize=12)
plt.grid()
plt.legend(loc='upper right')
#plt.savefig("Normalized Differential Cross-section Plot",dpi=600, bbox_inches="tight")
plt.show()
#%%----------------------------------------------------------------------------
#------------------------Cross-section Goodness of Fit:------------------------
#Thomson Goodness of Fit:
thomson_theory_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(norm_diff_cross_areas[2:],
                                                       get_thomson_fit_func(norm_diff_cross_areas_angles[2:], 20),
                                                       norm_diff_cross_areas_errs[2:], num_free_params=0)

print('\nReduced Chi-squared of Thomson Theory Fit:')
print(f'Reduced Chi-squared = {thomson_theory_rdd_chi_sqrd_val}')
    
#Klein-Nishina Goodness of Fit:
klein_theory_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(norm_diff_cross_areas[2:],
                                                       get_klein_fit_func(norm_diff_cross_areas_angles[2:], 20, E_gam), 
                                                       norm_diff_cross_areas_errs[2:], num_free_params=0)

print('\nReduced Chi-squared of Klein-Nishina Theory Fit:')
print(f'Reduced Chi-squared = {klein_theory_rdd_chi_sqrd_val}')
print('\n--------------------------------------------------------------------')
#%%----------------------------------------------------------------------------
#-----------------------------------Temp:--------------------------------------