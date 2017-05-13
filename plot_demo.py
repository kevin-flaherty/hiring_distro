def plot_demo():
    ''' Plot the distribution of time to faculty job for men/women '''

    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    with open('misc.csv') as fp:
        reader = csv.reader(fp,delimiter=',')
        next(reader,None)
        data_read = [row for row in reader]

    n = len(data_read)
    gender = []
    phd_year = []
    hire_year = []
    carnegie_class = []
    for i in range(n):
        if data_read[i][6] != 'X' and data_read[i][4] != '1900':
            if data_read[i][3] == 'M':
                gender.append(0)
            else:
                gender.append(1)
            #gender.append(data_read[i][3])
            phd_year.append(int(data_read[i][4]))
            hire_year.append(int(data_read[i][5]))
            if data_read[i][6] == 'R1':
                carnegie_class.append(0)
            else:
                carnegie_class.append(1)
            #carnegie_class.append(data_read[i][6])
    gender = np.array(gender)
    phd_year = np.array(phd_year)
    hire_year = np.array(hire_year)
    carnegie_class = np.array(carnegie_class)
            
    wmale = (gender == 0) & (phd_year > 2000)
    wfemale = (gender == 1) & (phd_year > 2000)
    wr1 = (carnegie_class == 0) & (phd_year > 2000)
    wnr1 = (carnegie_class == 1) & (phd_year > 2000)
    wmr1 = (gender == 0) & (carnegie_class == 0) & (phd_year > 2000)
    wfr1 = (gender == 1) & (carnegie_class == 0) & (phd_year > 2000)
    wmnr1 = (gender == 0) & (carnegie_class == 1) & (phd_year > 2000)
    wfnr1 = (gender == 1) & (carnegie_class ==1) & (phd_year > 2000)

    print 'Men: ',wmale.sum()
    print 'Women: ',wfemale.sum()
    print 'R1: ',wmr1.sum(),wfr1.sum()
    print 'Not R1: ',wmnr1.sum(),wfnr1.sum()

    bins=np.arange(-1,12)-.5
    p=plt.hist(hire_year[wmale]-phd_year[wmale],bins,color='k',lw=3,histtype='step')
    plt.axvline(np.mean(hire_year[wmale]-phd_year[wmale]),color='k',ls='--',lw=3)
    p=plt.hist(hire_year[wfemale]-phd_year[wfemale],bins,color='r',lw=3,histtype='step')
    plt.axvline(np.mean(hire_year[wfemale]-phd_year[wfemale]),color='r',ls='--',lw=3)

    print ' '
    print 'Time to faculty, Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmale]-phd_year[wmale]),np.mean(hire_year[wmale]-phd_year[wmale])/np.sqrt(wmale.sum()))
    print 'Time to faculty, Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfemale]-phd_year[wfemale]),np.mean(hire_year[wfemale]-phd_year[wfemale])/np.sqrt(wfemale.sum()))
    d,p = ks_2samp(hire_year[wmale]-phd_year[wmale],hire_year[wfemale]-phd_year[wfemale])
    print 'Prob: ',p

    print ' '
    print 'Time to faculty (R1), Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmr1]-phd_year[wmr1]),np.mean(hire_year[wmr1]-phd_year[wmr1])/np.sqrt(wmr1.sum()))
    print 'Time to faculty (R1), Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfr1]-phd_year[wfr1]),np.mean(hire_year[wfr1]-phd_year[wfr1])/np.sqrt(wfr1.sum()))
    d,p = ks_2samp(hire_year[wmr1]-phd_year[wmr1],hire_year[wfr1]-phd_year[wfr1])
    print 'Prob: ',p

    print ' '
    print 'Time to faculty (Not R1), Men: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wmnr1]-phd_year[wmnr1]),np.mean(hire_year[wmnr1]-phd_year[wmnr1])/np.sqrt(wmnr1.sum()))
    print 'Time to faculty (Not R1), Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfnr1]-phd_year[wfnr1]),np.mean(hire_year[wfnr1]-phd_year[wfnr1])/np.sqrt(wfnr1.sum()))
    d,p = ks_2samp(hire_year[wmnr1]-phd_year[wmnr1],hire_year[wfnr1]-phd_year[wfnr1])
    print 'Prob: ',p
