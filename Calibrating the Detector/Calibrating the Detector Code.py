import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_rdd_chi_sqrd_val(obs_array,calc_array,sigma_vals_array,num_free_params):
    "RETURNS: Reduced Chi-squared value for fitted gaussian."
    chi_sqrd_list = []
    num_obs = len(obs_array)
    obs_list = list(obs_array)
    calc_list = list(calc_array)
    sigma_vals_list = list(sigma_vals_array)
    for i in range(0,len(obs_list)):
        chi_sqrd_val = ((obs_list[i] - calc_list[i])**2) / (sigma_vals_list[i]**2)
        chi_sqrd_list.append(chi_sqrd_val)
    chi_sqrd = np.sum(chi_sqrd_list)
    rdd_chi_sqrd = chi_sqrd / (num_obs - num_free_params)
    return rdd_chi_sqrd

def get_sqr_rt_N_vals_fit(N_vals_fit_array):
    "RETURNS: Array of the square root of abs vals of N fit array."
    return_sqr_rt_N_vals_fit_list = []
    N_vals_fit_list = list(N_vals_fit_array)
    for N_val in N_vals_fit_list:
        sqr_rt_N_val = np.sqrt(np.maximum(N_val, 1))
        return_sqr_rt_N_vals_fit_list.append(sqr_rt_N_val)
    return_sqr_rt_N_vals_fit_array = np.array(return_sqr_rt_N_vals_fit_list)
    return return_sqr_rt_N_vals_fit_array
#------------------------Importing measured data:------------------------------
#We want to import all 800 V data:
#Bin Numbers Array for ALL Radiocative Sources:

n_vals = np.loadtxt( # Bin Number, n
    "Calibrating the Detector Data.csv", delimiter=',', usecols=8,
    skiprows=1, max_rows=256)

#Gaining Total Count Numbers for EACH Radioactive Source:
Am241_N_vals = np.loadtxt( # Total Count Number, N
    "Calibrating the Detector Data.csv", delimiter=',', usecols=9,
    skiprows=1, max_rows=256)

Cs137_N_vals = np.loadtxt( # Total Count Number, N
    "Calibrating the Detector Data.csv", delimiter=',', usecols=11,
    skiprows=1, max_rows=256)

Co57_N_vals = np.loadtxt( # Total Count Number, N
    "Calibrating the Detector Data.csv", delimiter=',', usecols=13,
    skiprows=1, max_rows=256)

Co60_N_vals = np.loadtxt( # Total Count Number, N
    "Calibrating the Detector Data.csv", delimiter=',', usecols=15,
    skiprows=1, max_rows=256)

Ba133_N_vals = np.loadtxt( # Total Count Number, N
    "Calibrating the Detector Data.csv", delimiter=',', usecols=17,
    skiprows=1, max_rows=256)
#%%----------------------------------------------------------------------------
#-------------------------------Plotting:--------------------------------------
#Want to gain Gaussian Fit and Standard Deviations:
def fit_gauss_func(n_domain, amp, mu, sigma, const, m_bg):
    '''Returns Array of Fitted N-values.'''
    gaus = amp*np.exp(-(n_domain-mu)**2 / (2 * sigma**2)) + const + m_bg*n_domain
    return gaus
#------------------------------------------------------------------------------
#Am-241:
plt.plot(n_vals,Am241_N_vals, color='red', marker=".", linestyle='',
         label="Am-241")

Am241_n_fit_domain_start = 8
Am241_n_fit_domain_end = 20

Am241_n_fit_domain = n_vals[Am241_n_fit_domain_start:Am241_n_fit_domain_end]

Am241_init_guesses = [60_000, 10, 2, 1_000, 0]

fit_Am241,cov_Am241 = curve_fit(fit_gauss_func, Am241_n_fit_domain,
                                Am241_N_vals[Am241_n_fit_domain_start:Am241_n_fit_domain_end],
                                Am241_init_guesses)

Am241_mu_err = np.sqrt(cov_Am241[1,1])  #The uncertainty in the mean.
Am241_sig_err = np.sqrt(cov_Am241[2,2]) #The uncertainty in the std deviation.

print('----------------------------------------------------------------------')
print('Am-241 Gaussian fit coefficients:')
print(fit_Am241)
print('\nAm-241 Covariance matrix:')
print(cov_Am241)
print('\nAm-241 Mean Bin Number and Error:')
print(f'mean n of peak energy = {fit_Am241[1]} +/- {Am241_mu_err}')
print('\nAm-241 Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_Am241[2]} +/- {Am241_sig_err}')

Am241_gauss_plot_domain = np.linspace(Am241_n_fit_domain_start, Am241_n_fit_domain_end, 10_000)
Am241_gauss_plot_vals = fit_gauss_func(Am241_gauss_plot_domain, amp=fit_Am241[0],
                                       mu=fit_Am241[1], sigma=fit_Am241[2],
                                       const=fit_Am241[3], m_bg=fit_Am241[4])

Am241_obs_vals = Am241_N_vals[Am241_n_fit_domain_start:Am241_n_fit_domain_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Am241_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Am241_obs_vals)

Am241_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Am241_obs_vals,
                                                fit_gauss_func(Am241_n_fit_domain,
                                                               fit_Am241[0], fit_Am241[1],
                                                               fit_Am241[2], fit_Am241[3],
                                                               m_bg=fit_Am241[4]),
                                                Am241_sqr_rt_N_vals_fit,num_free_params=5)

print('\nAm-241 Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Am241_rdd_chi_sqrd_val}')

plt.plot(Am241_gauss_plot_domain, Am241_gauss_plot_vals, 'black', alpha=0.8,
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
        
plt.xticks(np.arange(0, 30+2, 2))
plt.yticks(np.arange(0, 60_000+10_000, 10_000))
plt.xlim(0,30)
plt.ylim(0,60_000)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V Am-241 Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#Cs-137:
plt.plot(n_vals,Cs137_N_vals, color='red', marker=".", linestyle='',
         label="Cs-137")

Cs137_n_fit_domain_start = 80
Cs137_n_fit_domain_end = 105

Cs137_n_fit_domain = n_vals[Cs137_n_fit_domain_start:Cs137_n_fit_domain_end]

Cs137_init_guesses = [8_000, 92, 5, 500, -550]

fit_Cs137,cov_Cs137 = curve_fit(fit_gauss_func, Cs137_n_fit_domain,
                                Cs137_N_vals[Cs137_n_fit_domain_start:Cs137_n_fit_domain_end],
                                Cs137_init_guesses)

Cs137_mu_err = np.sqrt(cov_Cs137[1,1])  #The uncertainty in the mean.
Cs137_sig_err = np.sqrt(cov_Cs137[2,2]) #The uncertainty in the std deviation.

print('Cs-137 Gaussian fit coefficients:')
print(fit_Cs137)
print('\nCs-137 Covariance matrix:')
print(cov_Cs137)
print('\nCs-137 Mean Bin Number and Error:')
print(f'mean n of peak energy = {fit_Cs137[1]} +/- {Cs137_mu_err}')
print('\nCs-137 Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_Cs137[2]} +/- {Cs137_sig_err}')

Cs137_gauss_plot_domain = np.linspace(Cs137_n_fit_domain_start, Cs137_n_fit_domain_end, 10_000)
Cs137_gauss_plot_vals = fit_gauss_func(Cs137_gauss_plot_domain, amp=fit_Cs137[0],
                                        mu=fit_Cs137[1], sigma=fit_Cs137[2],
                                        const=fit_Cs137[3], m_bg=fit_Cs137[4])

Cs137_obs_vals = Cs137_N_vals[Cs137_n_fit_domain_start:Cs137_n_fit_domain_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Cs137_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Cs137_obs_vals)

Cs137_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Cs137_obs_vals,
                                                fit_gauss_func(Cs137_n_fit_domain, fit_Cs137[0],
                                                               fit_Cs137[1], fit_Cs137[2],
                                                               fit_Cs137[3], fit_Cs137[4]),
                                                Cs137_sqr_rt_N_vals_fit,num_free_params=5)

print('\nCs-137 Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Cs137_rdd_chi_sqrd_val}')

plt.plot(Cs137_gauss_plot_domain, Cs137_gauss_plot_vals, 'black', alpha=0.8,
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
        
plt.xticks(np.arange(0, 120+10, 10))
plt.yticks(np.arange(0, 8_000+1_000, 1_000))
plt.xlim(0,120)
plt.ylim(0,8_000)
plt.xlabel("Bin Number, n")
plt.ylabel("Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V Cs-137 Gaussian Counts Plot",dpi=600, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#Co-57:
plt.plot(n_vals,Co57_N_vals, color='red', marker=".", linestyle='',
         label="Co-57")

Co57_n_fit_domain_start = 10
Co57_n_fit_domain_end = 23

Co57_n_fit_domain = n_vals[Co57_n_fit_domain_start:Co57_n_fit_domain_end]

Co57_init_guesses = [50, 18, 2, 20, -2]

fit_Co57,cov_Co57 = curve_fit(fit_gauss_func, Co57_n_fit_domain,
                                Co57_N_vals[Co57_n_fit_domain_start:Co57_n_fit_domain_end],
                                Co57_init_guesses)

Co57_mu_err = np.sqrt(cov_Co57[1,1])  #The uncertainty in the mean.
Co57_sig_err = np.sqrt(cov_Co57[2,2]) #The uncertainty in the std deviation.

print('Co-57 Gaussian fit coefficients:')
print(fit_Co57)
print('\nCo-57 Covariance matrix:')
print(cov_Co57)
print('\nCo-57 Mean Bin Number and Error:')
print(f'mean n of peak energy = {fit_Co57[1]} +/- {Co57_mu_err}')
print('\nCo-57 Standard Deviation and Error:')
print(f'sigma of gaussian = {fit_Co57[2]} +/- {Co57_sig_err}')

Co57_gauss_plot_domain = np.linspace(Co57_n_fit_domain_start, Co57_n_fit_domain_end, 10_000)
Co57_gauss_plot_vals = fit_gauss_func(Co57_gauss_plot_domain, amp=fit_Co57[0],
                                        mu=fit_Co57[1], sigma=fit_Co57[2],
                                        const=fit_Co57[3], m_bg=fit_Co57[4])

Co57_obs_vals = Co57_N_vals[Co57_n_fit_domain_start:Co57_n_fit_domain_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Co57_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Co57_obs_vals)

Co57_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Co57_obs_vals,
                                                fit_gauss_func(Co57_n_fit_domain,
                                                               fit_Co57[0], fit_Co57[1],
                                                               fit_Co57[2], fit_Co57[3],
                                                               fit_Co57[4]),
                                                Co57_sqr_rt_N_vals_fit,num_free_params=5)

print('\nCo-57 Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Co57_rdd_chi_sqrd_val}')

plt.plot(Co57_gauss_plot_domain, Co57_gauss_plot_vals, 'black', alpha=0.8,
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
        
plt.xticks(np.arange(0, 120+10, 10))
plt.yticks(np.arange(0, 90+10, 10))
plt.xlim(0,120)
plt.ylim(0,90)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V Co-57 Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#Co-60:
plt.plot(n_vals,Co60_N_vals, color='red', marker=".", linestyle='',
         label="Co-60")

Co60_n_fit_domain_lft_pk_start = 152
Co60_n_fit_domain_lft_pk_end = 168

Co60_n_fit_domain_lft_pk = n_vals[Co60_n_fit_domain_lft_pk_start:Co60_n_fit_domain_lft_pk_end]

Co60_init_guesses_lft_pk = [375, 160, 3, 100, -4]

fit_Co60_lft_pk,cov_Co60_lft_pk = curve_fit(fit_gauss_func, Co60_n_fit_domain_lft_pk,
                                Co60_N_vals[Co60_n_fit_domain_lft_pk_start:Co60_n_fit_domain_lft_pk_end],
                                Co60_init_guesses_lft_pk)

Co60_mu_err_lft_pk = np.sqrt(cov_Co60_lft_pk[1,1])  #The uncertainty in the mean.
Co60_sig_err_lft_pk = np.sqrt(cov_Co60_lft_pk[2,2]) #The uncertainty in the std deviation.

print('Co-60 Left Energy Peak Gaussian fit coefficients:')
print(fit_Co60_lft_pk)
print('\nCo-60 Left Energy Peak Covariance matrix:')
print(cov_Co60_lft_pk)
print('\nCo-60 Left Peak Mean Bin Number and Error:')
print(f'mean n of left peak energy = {fit_Co60_lft_pk[1]} +/- {Co60_mu_err_lft_pk}')
print('\nCo-60 Left Peak Standard Deviation and Error:')
print(f'sigma of left peak gaussian = {fit_Co60_lft_pk[2]} +/- {Co60_sig_err_lft_pk}')

Co60_gauss_plot_domain_lft_pk = np.linspace(Co60_n_fit_domain_lft_pk_start, Co60_n_fit_domain_lft_pk_end, 10_000)
Co60_gauss_plot_vals_lft_pk = fit_gauss_func(Co60_gauss_plot_domain_lft_pk,
                                             amp=fit_Co60_lft_pk[0],
                                             mu=fit_Co60_lft_pk[1],
                                             sigma=fit_Co60_lft_pk[2],
                                             const=fit_Co60_lft_pk[3],
                                             m_bg=fit_Co60_lft_pk[4])

Co60_obs_vals_lft_pk = Co60_N_vals[Co60_n_fit_domain_lft_pk_start:Co60_n_fit_domain_lft_pk_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Co60_lft_pk_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Co60_obs_vals_lft_pk)

Co60_lft_pk_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Co60_obs_vals_lft_pk,
                                                    fit_gauss_func(Co60_n_fit_domain_lft_pk,
                                                                   fit_Co60_lft_pk[0],
                                                                   fit_Co60_lft_pk[1],
                                                                   fit_Co60_lft_pk[2],
                                                                   fit_Co60_lft_pk[3],
                                                                   fit_Co60_lft_pk[4]),
                                                    Co60_lft_pk_sqr_rt_N_vals_fit,num_free_params=5)

print('\nCo-60 Left Peak Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Co60_lft_pk_rdd_chi_sqrd_val}')

plt.plot(Co60_gauss_plot_domain_lft_pk, Co60_gauss_plot_vals_lft_pk, 'black',
         alpha=0.8, linestyle='dashed', label='Gaussian Fit')

Co60_n_fit_domain_rgt_pk_start = 170
Co60_n_fit_domain_rgt_pk_end = 190

Co60_n_fit_domain_rgt_pk = n_vals[Co60_n_fit_domain_rgt_pk_start:Co60_n_fit_domain_rgt_pk_end]

Co60_init_guesses_rgt_pk = [300, 180, 3, 50, -2]

fit_Co60_rgt_pk,cov_Co60_rgt_pk = curve_fit(fit_gauss_func, Co60_n_fit_domain_rgt_pk,
                                Co60_N_vals[Co60_n_fit_domain_rgt_pk_start:Co60_n_fit_domain_rgt_pk_end],
                                Co60_init_guesses_rgt_pk)

Co60_mu_err_rgt_pk = np.sqrt(cov_Co60_rgt_pk[1,1])  #The uncertainty in the mean.
Co60_sig_err_rgt_pk = np.sqrt(cov_Co60_rgt_pk[2,2]) #The uncertainty in the std deviation.
print('......................................................................')
print('\nCo-60 Right Energy Peak Gaussian fit coefficients:')
print(fit_Co60_rgt_pk)
print('\nCo-60 Right Energy Peak Covariance matrix:')
print(cov_Co60_rgt_pk)
print('\nCo-60 Right Peak Mean Bin Number and Error:')
print(f'mean n of right peak energy = {fit_Co60_rgt_pk[1]} +/- {Co60_mu_err_rgt_pk}')
print('\nCo-60 Right Peak Standard Deviation and Error:')
print(f'sigma of right peak gaussian = {fit_Co60_rgt_pk[2]} +/- {Co60_sig_err_rgt_pk}')

Co60_gauss_plot_domain_rgt_pk = np.linspace(Co60_n_fit_domain_rgt_pk_start, Co60_n_fit_domain_rgt_pk_end, 10_000)
Co60_gauss_plot_vals_rgt_pk = fit_gauss_func(Co60_gauss_plot_domain_rgt_pk,
                                             amp=fit_Co60_rgt_pk[0],
                                             mu=fit_Co60_rgt_pk[1],
                                             sigma=fit_Co60_rgt_pk[2],
                                             const=fit_Co60_rgt_pk[3],
                                             m_bg=fit_Co60_rgt_pk[4])

Co60_obs_vals_rgt_pk = Co60_N_vals[Co60_n_fit_domain_rgt_pk_start:Co60_n_fit_domain_rgt_pk_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Co60_rgt_pk_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Co60_obs_vals_rgt_pk)

Co60_rgt_pk_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Co60_obs_vals_rgt_pk,
                                                    fit_gauss_func(Co60_n_fit_domain_rgt_pk,
                                                                   fit_Co60_rgt_pk[0],
                                                                   fit_Co60_rgt_pk[1],
                                                                   fit_Co60_rgt_pk[2],
                                                                   fit_Co60_rgt_pk[3],
                                                                   fit_Co60_rgt_pk[4]),
                                                    Co60_rgt_pk_sqr_rt_N_vals_fit,num_free_params=5)

print('\nCo-60 Right Peak Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Co60_rgt_pk_rdd_chi_sqrd_val}')

plt.plot(Co60_gauss_plot_domain_rgt_pk, Co60_gauss_plot_vals_rgt_pk, 'blue',
         alpha=0.8, linestyle='dashed', label='Gaussian Fit')

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
        
plt.xticks(np.arange(100, 200+10, 10))
plt.yticks(np.arange(0, 400+50, 50))
plt.xlim(100,200)
plt.ylim(0,400)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend(loc='upper left')
#plt.savefig("800 V Co-60 Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#------------------------------------------------------------------------------
#Ba-133:
plt.plot(n_vals,Ba133_N_vals, color='red', marker=".", linestyle='',
         label="Ba-133")

Ba133_n_fit_domain_lft_pk_start = 1
Ba133_n_fit_domain_lft_pk_end = 8

Ba133_n_fit_domain_lft_pk = n_vals[Ba133_n_fit_domain_lft_pk_start:Ba133_n_fit_domain_lft_pk_end]

Ba133_init_guesses_lft_pk = [5_000, 5, 1, 250, 0]

fit_Ba133_lft_pk,cov_Ba133_lft_pk = curve_fit(fit_gauss_func, Ba133_n_fit_domain_lft_pk,
                                Ba133_N_vals[Ba133_n_fit_domain_lft_pk_start:Ba133_n_fit_domain_lft_pk_end],
                                Ba133_init_guesses_lft_pk)

Ba133_mu_err_lft_pk = np.sqrt(cov_Ba133_lft_pk[1,1])  #The uncertainty in the mean.
Ba133_sig_err_lft_pk = np.sqrt(cov_Ba133_lft_pk[2,2]) #The uncertainty in the std deviation.

print('Ba-133 Left Energy Peak Gaussian fit coefficients:')
print(fit_Ba133_lft_pk)
print('\nBa-133 Left Energy Peak Covariance matrix:')
print(cov_Ba133_lft_pk)
print('\nBa-133 Left Peak Mean Bin Number and Error:')
print(f'mean n of left peak energy = {fit_Ba133_lft_pk[1]} +/- {Ba133_mu_err_lft_pk}')
print('\nBa-133 Left Peak Standard Deviation and Error:')
print(f'sigma of left peak gaussian = {fit_Ba133_lft_pk[2]} +/- {Ba133_sig_err_lft_pk}')

Ba133_gauss_plot_domain_lft_pk = np.linspace(Ba133_n_fit_domain_lft_pk_start, Ba133_n_fit_domain_lft_pk_end, 10_000)
Ba133_gauss_plot_vals_lft_pk = fit_gauss_func(Ba133_gauss_plot_domain_lft_pk,
                                             amp=fit_Ba133_lft_pk[0],
                                             mu=fit_Ba133_lft_pk[1],
                                             sigma=fit_Ba133_lft_pk[2],
                                             const=fit_Ba133_lft_pk[3],
                                             m_bg=fit_Ba133_lft_pk[4])

Ba133_obs_vals_lft_pk = Ba133_N_vals[Ba133_n_fit_domain_lft_pk_start:Ba133_n_fit_domain_lft_pk_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Ba133_lft_pk_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Ba133_obs_vals_lft_pk)

Ba133_lft_pk_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Ba133_obs_vals_lft_pk,
                                                    fit_gauss_func(Ba133_n_fit_domain_lft_pk,
                                                                   fit_Ba133_lft_pk[0],
                                                                   fit_Ba133_lft_pk[1],
                                                                   fit_Ba133_lft_pk[2],
                                                                   fit_Ba133_lft_pk[3],
                                                                   fit_Ba133_lft_pk[4]),
                                                    Ba133_lft_pk_sqr_rt_N_vals_fit,num_free_params=5)

print('\nBa-133 Left Peak Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Ba133_lft_pk_rdd_chi_sqrd_val}')

plt.plot(Ba133_gauss_plot_domain_lft_pk, Ba133_gauss_plot_vals_lft_pk, 'black',
         alpha=0.8, linestyle='dashed', label='Gaussian Fit')

Ba133_n_fit_domain_mid_pk_start = 10
Ba133_n_fit_domain_mid_pk_end = 17

Ba133_n_fit_domain_mid_pk = n_vals[Ba133_n_fit_domain_mid_pk_start:Ba133_n_fit_domain_mid_pk_end]

Ba133_init_guesses_mid_pk = [1_750, 13, 1, 250, 0]

fit_Ba133_mid_pk,cov_Ba133_mid_pk = curve_fit(fit_gauss_func, Ba133_n_fit_domain_mid_pk,
                                Ba133_N_vals[Ba133_n_fit_domain_mid_pk_start:Ba133_n_fit_domain_mid_pk_end],
                                Ba133_init_guesses_mid_pk)

Ba133_mu_err_mid_pk = np.sqrt(cov_Ba133_mid_pk[1,1])  #The uncertainty in the mean.
Ba133_sig_err_mid_pk = np.sqrt(cov_Ba133_mid_pk[2,2]) #The uncertainty in the std deviation.

print('......................................................................')
print('Ba-133 Middle Energy Peak Gaussian fit coefficients:')
print(fit_Ba133_mid_pk)
print('\nBa-133 Middle Energy Peak Covariance matrix:')
print(cov_Ba133_mid_pk)
print('\nBa-133 Middle Peak Mean Bin Number and Error:')
print(f'mean n of middle peak energy = {fit_Ba133_mid_pk[1]} +/- {Ba133_mu_err_mid_pk}')
print('\nBa-133 Middle Peak Standard Deviation and Error:')
print(f'sigma of middle peak gaussian = {fit_Ba133_mid_pk[2]} +/- {Ba133_sig_err_mid_pk}')

Ba133_gauss_plot_domain_mid_pk = np.linspace(Ba133_n_fit_domain_mid_pk_start, Ba133_n_fit_domain_mid_pk_end, 10_000)
Ba133_gauss_plot_vals_mid_pk = fit_gauss_func(Ba133_gauss_plot_domain_mid_pk,
                                             amp=fit_Ba133_mid_pk[0],
                                             mu=fit_Ba133_mid_pk[1],
                                             sigma=fit_Ba133_mid_pk[2],
                                             const=fit_Ba133_mid_pk[3],
                                             m_bg=fit_Ba133_mid_pk[4])

Ba133_obs_vals_mid_pk = Ba133_N_vals[Ba133_n_fit_domain_mid_pk_start:Ba133_n_fit_domain_mid_pk_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Ba133_mid_pk_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Ba133_obs_vals_mid_pk)

Ba133_mid_pk_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Ba133_obs_vals_mid_pk,
                                                    fit_gauss_func(Ba133_n_fit_domain_mid_pk,
                                                                   fit_Ba133_mid_pk[0],
                                                                   fit_Ba133_mid_pk[1],
                                                                   fit_Ba133_mid_pk[2],
                                                                   fit_Ba133_mid_pk[3],
                                                                   fit_Ba133_mid_pk[4]),
                                                    Ba133_mid_pk_sqr_rt_N_vals_fit,num_free_params=5)

print('\nBa-133 Middle Peak Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Ba133_mid_pk_rdd_chi_sqrd_val}')

plt.plot(Ba133_gauss_plot_domain_mid_pk, Ba133_gauss_plot_vals_mid_pk, 'green',
         alpha=0.8, linestyle='dashed', label='Gaussian Fit')

Ba133_n_fit_domain_rgt_pk_start = 47
Ba133_n_fit_domain_rgt_pk_end = 55

Ba133_n_fit_domain_rgt_pk = n_vals[Ba133_n_fit_domain_rgt_pk_start:Ba133_n_fit_domain_rgt_pk_end]

Ba133_init_guesses_rgt_pk = [1_400, 51, 3, 250, 0]

fit_Ba133_rgt_pk,cov_Ba133_rgt_pk = curve_fit(fit_gauss_func, Ba133_n_fit_domain_rgt_pk,
                                Ba133_N_vals[Ba133_n_fit_domain_rgt_pk_start:Ba133_n_fit_domain_rgt_pk_end],
                                Ba133_init_guesses_rgt_pk)

Ba133_mu_err_rgt_pk = np.sqrt(cov_Ba133_rgt_pk[1,1])  #The uncertainty in the mean.
Ba133_sig_err_rgt_pk = np.sqrt(cov_Ba133_rgt_pk[2,2]) #The uncertainty in the std deviation.

print('......................................................................')
print('Ba-133 Right Energy Peak Gaussian fit coefficients:')
print(fit_Ba133_rgt_pk)
print('\nBa-133 Right Energy Peak Covariance matrix:')
print(cov_Ba133_rgt_pk)
print('\nBa-133 Right Peak Mean Bin Number and Error:')
print(f'mean n of right peak energy = {fit_Ba133_rgt_pk[1]} +/- {Ba133_mu_err_rgt_pk}')
print('\nBa-133 Right Peak Standard Deviation and Error:')
print(f'sigma of right peak gaussian = {fit_Ba133_rgt_pk[2]} +/- {Ba133_sig_err_rgt_pk}')

Ba133_gauss_plot_domain_rgt_pk = np.linspace(Ba133_n_fit_domain_rgt_pk_start, Ba133_n_fit_domain_rgt_pk_end, 10_000)
Ba133_gauss_plot_vals_rgt_pk = fit_gauss_func(Ba133_gauss_plot_domain_rgt_pk,
                                             amp=fit_Ba133_rgt_pk[0],
                                             mu=fit_Ba133_rgt_pk[1],
                                             sigma=fit_Ba133_rgt_pk[2],
                                             const=fit_Ba133_rgt_pk[3],
                                             m_bg=fit_Ba133_rgt_pk[4])

Ba133_obs_vals_rgt_pk = Ba133_N_vals[Ba133_n_fit_domain_rgt_pk_start:Ba133_n_fit_domain_rgt_pk_end]
#For Chi-squared test: this is our sigma array NOT FOR GAUSSIAN.
Ba133_rgt_pk_sqr_rt_N_vals_fit = get_sqr_rt_N_vals_fit(Ba133_obs_vals_rgt_pk)

Ba133_rgt_pk_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val(Ba133_obs_vals_rgt_pk,
                                                    fit_gauss_func(Ba133_n_fit_domain_rgt_pk,
                                                                   fit_Ba133_rgt_pk[0],
                                                                   fit_Ba133_rgt_pk[1],
                                                                   fit_Ba133_rgt_pk[2],
                                                                   fit_Ba133_rgt_pk[3],
                                                                   fit_Ba133_rgt_pk[4]),
                                                    Ba133_rgt_pk_sqr_rt_N_vals_fit,num_free_params=5)

print('\nBa-133 Right Peak Reduced Chi-squared of Fitted Gaussian:')
print(f'Reduced Chi-squared = {Ba133_rgt_pk_rdd_chi_sqrd_val}')

plt.plot(Ba133_gauss_plot_domain_rgt_pk, Ba133_gauss_plot_vals_rgt_pk, 'blue',
         alpha=0.8, linestyle='dashed', label='Gaussian Fit')

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
        
plt.xticks(np.arange(0, 65+5, 5))
plt.yticks(np.arange(0, 6_000+500, 500))
plt.xlim(0,65)
plt.ylim(0,6_000)
plt.xlabel("Bin Number, n")
plt.ylabel("Total Counts, N")
plt.grid()
plt.legend()
#plt.savefig("800 V Ba-133 Gaussian Counts Plot",dpi=1200, bbox_inches="tight")
plt.show()
print('\n--------------------------------------------------------------------')
#%%----------------------------------------------------------------------------
#-------------------------------Plotting:--------------------------------------
#Putting all sigmas into ordered array:
sigma_vals = [fit_Am241[2],fit_Cs137[2], fit_Co57[2], fit_Co60_lft_pk[2],
              fit_Co60_rgt_pk[2],fit_Ba133_lft_pk[2],fit_Ba133_mid_pk[2],
              fit_Ba133_rgt_pk[2]]

sigma_vals = np.array(sigma_vals)
#------------------------------------------------------------------------------
#Putting all peak n-values into ordered array:
pk_n_vals = [fit_Am241[1],fit_Cs137[1], fit_Co57[1], fit_Co60_lft_pk[1],
              fit_Co60_rgt_pk[1],fit_Ba133_lft_pk[1],fit_Ba133_mid_pk[1],
              fit_Ba133_rgt_pk[1]]

pk_n_errs = [Am241_mu_err,Cs137_mu_err,Co57_mu_err, Co60_mu_err_lft_pk,
             Co60_mu_err_rgt_pk,Ba133_mu_err_lft_pk,Ba133_mu_err_mid_pk,
             Ba133_mu_err_rgt_pk]

pk_n_errs = np.array(pk_n_errs)
#------------------------------------------------------------------------------
#Putting corresponding known photon energies into ordered array:
pk_n_energies = [59.54, 661.7, 122.0, 1173.2, 1332.5, 30.85, 81.0, 356.0] #in keV
pk_n_energies = np.array(pk_n_energies)

pk_n_energies_errs = [0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.1, 0.1] #in keV
pk_n_energies_errs = np.array(pk_n_energies_errs)
#------------------------------------------------------------------------------
def get_line_func(gradient, intercept, x_domain_array):
    "RETURNS: Array of fitted photon energy values."
    return_line_func_list = []
    x_domain_list = list(x_domain_array)
    for x_val in x_domain_list:
        line_func_val = (gradient*x_val) + intercept
        return_line_func_list.append(line_func_val)
    return_line_func_array = np.array(return_line_func_list)
    return return_line_func_array

def get_rdd_chi_sqrd_val_linear(obs_array,calc_array,sigma_vals_array,num_free_params):
    "RETURNS: Reduced Chi-squared value for fitted linear regression."
    chi2 = np.sum(((obs_array - calc_array)**2) / (sigma_vals_array**2))
    dof = len(list(obs_array)) - num_free_params
    rdd_chi_sqrd = chi2 / dof
    return rdd_chi_sqrd
#------------------------------------------------------------------------------
plt.errorbar(pk_n_vals,pk_n_energies, color='red', marker=".",
         linestyle='', label="Experiment Data", yerr=pk_n_energies_errs, capsize=4,
         xerr=pk_n_errs, ecolor='black')

n_fit_domain = np.linspace(0, 200, 10_000)

fit_energy,cov_energy = np.polyfit(pk_n_vals,pk_n_energies, 1, w=1/pk_n_errs,
                                   cov=True)

print('Photon Energy fit coefficients:')
print(fit_energy)
print('\nPhoton Energy covariance matrix:')
print(cov_energy)
print('\nPhoton Energy polynomial:')
print(f'Photon Energy = {fit_energy[0]}(n) + {fit_energy[1]}')

energy_slope_err = np.sqrt(cov_energy[0,0]) #The uncertainty in the slope ('gradient')
energy_intercept_err = np.sqrt(cov_energy[1,1]) #The uncertainty in the intercept.

print('\nEnergy gradient and error:')
print(f'gradient = {fit_energy[0]} +/- {energy_slope_err}')
print('\nEnergy intercept and error:')
print(f'intercept = {fit_energy[1]} +/- {energy_intercept_err}')

fit_energy_vals = get_line_func(fit_energy[0],fit_energy[1],n_fit_domain)

#For Chi-squared test:
energy_poly_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val_linear(pk_n_energies,
                                                           get_line_func(fit_energy[0],
                                                                         fit_energy[1],
                                                                         pk_n_vals),
                                                           pk_n_errs*fit_energy[0],num_free_params=2)

print('\nPhoton Energy Reduced Chi-squared of Fitted Linear Polynomial:')
print(f'Reduced Chi-squared = {energy_poly_rdd_chi_sqrd_val}')
#High because NOT considering more prevelant error in bin number.

#Introduce some minimum sigma floor to account for real experimental scatter:
pk_n_err_floor = 2  # bins
pk_n_errs = np.maximum(pk_n_errs, pk_n_err_floor)

energy_poly_rdd_chi_sqrd_val = get_rdd_chi_sqrd_val_linear(pk_n_energies,
                                                           get_line_func(fit_energy[0],
                                                                         fit_energy[1],
                                                                         pk_n_vals),
                                                           pk_n_errs*fit_energy[0],num_free_params=2)

print(f'\nNew Reduced Chi-squared (after flooring uncertainties)= {energy_poly_rdd_chi_sqrd_val}')

plt.plot(n_fit_domain, fit_energy_vals, 'blue', alpha=0.8, linestyle='dashed',
         label='Linear Fit')

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
        
plt.xticks(np.arange(0, 200+20, 20))
plt.yticks(np.arange(0, 1_400+100, 100))
plt.xlim(0,200)
plt.ylim(0,1_400)
plt.xlabel("Bin Number, n")
plt.ylabel("Photon Energy (keV)")
plt.grid()
plt.legend()
#plt.savefig("800 V Calibration Plot",dpi=600, bbox_inches="tight")
plt.show()
#%%----------------------------------------------------------------------------
#-----------------------------------Temp:--------------------------------------
