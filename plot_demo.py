def load_demo():
    '''Load in the hiring information into appropriate variables'''
    import csv
    import numpy as np
    
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
        if data_read[i][3] != 'X' and data_read[i][1] != '1900':
            if data_read[i][0] == 'M':
                gender.append(0)
            else:
                gender.append(1)
            phd_year.append(int(data_read[i][1]))
            hire_year.append(int(data_read[i][2]))
            if data_read[i][3] == 'R1':
                carnegie_class.append(0)
            else:
                carnegie_class.append(1)
    gender = np.array(gender)
    phd_year = np.array(phd_year)
    hire_year = np.array(hire_year)
    carnegie_class = np.array(carnegie_class)

    return gender,phd_year,hire_year,carnegie_class

def plot_demo(full=False):
    ''' Plot the distribution of time to faculty job for men/women '''

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    gender,phd_year,hire_year,carnegie_class = load_demo()       
    wmale = (gender == 0) & (phd_year > 2000)
    wfemale = (gender == 1) & (phd_year > 2000)
    wr1 = (carnegie_class == 0) & (phd_year > 2000)
    wnr1 = (carnegie_class == 1) & (phd_year > 2000)
    wmr1 = (gender == 0) & (carnegie_class == 0) & (phd_year > 2000)
    wfr1 = (gender == 1) & (carnegie_class == 0) & (phd_year > 2000)
    wmnr1 = (gender == 0) & (carnegie_class == 1) & (phd_year > 2000)
    wfnr1 = (gender == 1) & (carnegie_class ==1) & (phd_year > 2000)
    wuse = phd_year>2000

    #print 'Men: ',wmale.sum() #137
    #print 'Women: ',wfemale.sum() #78
    #print 'R1: ',wmr1.sum(),wfr1.sum() #106, 52
    #print 'Not R1: ',wmnr1.sum(),wfnr1.sum() #31, 26

    bins=np.arange(-1,11)-.5
    if full:
        #Plot full distribution
        plt.rc('axes',lw=3)
        p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step')
        plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',ls='--',lw=3)
        plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=24)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        plt.tight_layout()
        
        print ' '
        print 'Mean time to hiring: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wuse]-phd_year[wuse]),np.mean(hire_year[wuse]-phd_year[wuse])/np.sqrt(wuse.sum()))
    else:
        #Plot distribution according to gender
    
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
