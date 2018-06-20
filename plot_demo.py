# - calculate the fraction of people that receive a PhD in a given year that eventually get a faculty job. Does it match the survival statistics?

def load_demo(file='misc.csv'):
    '''Load in the hiring information into appropriate variables'''
    import csv
    import numpy as np

    #file = misc.csv: hiring statistics from rumor mill
    #file = misc2.csv: hiring statistics from select departments
    
    with open(file) as fp:
        reader = csv.reader(fp,delimiter=',')
        next(reader,None)
        data_read = [row for row in reader]

    n = len(data_read)
    gender = []
    phd_year = []
    hire_year = []
    carnegie_class = []

    for i in range(n):
        if data_read[i][3] != 'X' and data_read[i][1] != '1900' and data_read[i][2]!='X' and data_read[i][1]!='X':
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

def plot_demo(file='misc.csv',full=False):
    ''' Plot the distribution of time to faculty job for men/women '''

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    gender,phd_year,hire_year,carnegie_class = load_demo(file)       
    wmale = (gender == 0) & (phd_year > 2000)
    wfemale = (gender == 1) & (phd_year > 2000)
    wr1 = (carnegie_class == 0) & (phd_year > 2000)
    wnr1 = (carnegie_class == 1) & (phd_year > 2000)
    wmr1 = (gender == 0) & (carnegie_class == 0) & (phd_year > 2000)
    wfr1 = (gender == 1) & (carnegie_class == 0) & (phd_year > 2000)
    wmnr1 = (gender == 0) & (carnegie_class == 1) & (phd_year > 2000)
    wfnr1 = (gender == 1) & (carnegie_class ==1) & (phd_year > 2000)
    wuse = phd_year>2000

    #print 'Men: ',wmale.sum() #157
    #print 'Women: ',wfemale.sum() #88
    #print 'R1: ',wmr1.sum(),wfr1.sum() #124, 58
    #print 'Not R1: ',wmnr1.sum(),wfnr1.sum() #33, 30
    #print np.mean(hire_year[wmr1]-phd_year[wmr1]),np.mean(hire_year[wfr1]-phd_year[wfr1]) #5.31, 4.13
    #print np.mean(hire_year[wmnr1]-phd_year[wmnr1]),np.mean(hire_year[wfnr1]-phd_year[wfnr1]) #5.30, 4.2
    
    bins=np.arange(0,11)-.5
    if full:
        #Plot full distribution
        plt.rc('axes',lw=2)
        p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0.,elinewidth=3)
        plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',ls='--',lw=3)
        plt.xlabel('Time to hiring (years)',fontsize=20)
        plt.ylabel('N',fontsize=24)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        
        print ' '
        print 'Mean time to hiring: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wuse]-phd_year[wuse]),np.mean(hire_year[wuse]-phd_year[wuse])/np.sqrt(wuse.sum()))
        #print 'Total sample size: {:0.0f}'.format(wuse.sum())#215
    else:
        #Plot distribution according to gender
        plt.rc('axes',lw=2)
        p=plt.hist(hire_year[wmale]-phd_year[wmale],bins,color='k',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0.,elinewidth=3)
        plt.axvline(np.mean(hire_year[wmale]-phd_year[wmale]),color='k',lw=3,ls=':')
        p=plt.hist(hire_year[wfemale]-phd_year[wfemale],bins,color='r',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='or',capthick=0.,elinewidth=3)
        plt.axvline(np.mean(hire_year[wfemale]-phd_year[wfemale]),color='r',lw=3,ls=':')
        plt.xlabel('Time to hiring (years)',fontsize=20)
        plt.ylabel('N',fontsize=20)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        plt.legend(('Male','Female'),loc='upper left',fontsize=16,frameon=False)

        #print 'Sample size (Male,Female): ',wmale.sum(),wfemale.sum()#137,78
        print ' '
        print 'Time to faculty, Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmale]-phd_year[wmale]),np.mean(hire_year[wmale]-phd_year[wmale])/np.sqrt(wmale.sum()))
        print 'Time to faculty, Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfemale]-phd_year[wfemale]),np.mean(hire_year[wfemale]-phd_year[wfemale])/np.sqrt(wfemale.sum()))
        d,p = ks_2samp(hire_year[wmale]-phd_year[wmale],hire_year[wfemale]-phd_year[wfemale])
        print 'Prob: {:0.3f}'.format(p)

        #print ' '
        #print 'Time to faculty (R1), Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmr1]-phd_year[wmr1]),np.mean(hire_year[wmr1]-phd_year[wmr1])/np.sqrt(wmr1.sum()))
        #print 'Time to faculty (R1), Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfr1]-phd_year[wfr1]),np.mean(hire_year[wfr1]-phd_year[wfr1])/np.sqrt(wfr1.sum()))
        #d,p = ks_2samp(hire_year[wmr1]-phd_year[wmr1],hire_year[wfr1]-phd_year[wfr1])
        #print 'Prob: ',p

        #print ' '
        #print 'Time to faculty (Not R1), Men: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wmnr1]-phd_year[wmnr1]),np.mean(hire_year[wmnr1]-phd_year[wmnr1])/np.sqrt(wmnr1.sum()))
        #print 'Time to faculty (Not R1), Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfnr1]-phd_year[wfnr1]),np.mean(hire_year[wfnr1]-phd_year[wfnr1])/np.sqrt(wfnr1.sum()))
        #d,p = ks_2samp(hire_year[wmnr1]-phd_year[wmnr1],hire_year[wfnr1]-phd_year[wfnr1])
        #print 'Prob: ',p


def plot_demo_split(file='misc.csv',full=False):
    '''Similar to plot_demo, but it plots the hiring time distribution in two year bins to look for any dramatic changes with time.'''

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    #2011,2012; 2013,2014; 2015,2016; 2017
    gender,phd_year,hire_year,carnegie_class = load_demo(file)
    bins = np.arange(0,11)-.5
    
    if full:
        for year in np.arange(2011,2018,2):
            wuse = (phd_year>2000) & (hire_year>=year) & (hire_year <= year+1)
            plt.subplot(4,1,(year-2011)/2+1)
            p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,lw=2,histtype='step',color='k')
            plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0.,elinewidth=2)
            plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',ls='--',lw=3)
            print 'Median time to hiring for {:0.0f}-{:0.0f}: {:0.2f} +- {:0.2f}'.format(year,year+1,np.mean(hire_year[wuse]-phd_year[wuse]),np.mean(hire_year[wuse]-phd_year[wuse])/np.sqrt(wuse.sum()))
            plt.title('{:0.0f} - {:0.0f}'.format(year,year+1),fontsize=16)
            ax = plt.gca()
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
        plt.xlabel('Time to hiring (years)',fontsize=18)
        plt.ylabel('N',fontsize=18)
    else:
        #look at 2011-2013 and 2014-2017
        wmale = (phd_year>2000) & (hire_year>=2011) & (hire_year<=2013) & (gender==0)
        wfemale = (phd_year>2000) & (hire_year>=2011) & (hire_year<=2013) & (gender==1)
        plt.subplot(121)
        p=plt.hist(hire_year[wmale]-phd_year[wmale],bins,color='k',lw=2,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0.,elinewidth=2)
        plt.axvline(np.mean(hire_year[wmale]-phd_year[wmale]),color='k',lw=2,ls=':')
        p=plt.hist(hire_year[wfemale]-phd_year[wfemale],bins,color='r',lw=2,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='or',capthick=0.,elinewidth=2)
        plt.axvline(np.mean(hire_year[wfemale]-phd_year[wfemale]),color='r',lw=2,ls=':')
        plt.xlabel('Time to hiring (years)',fontsize=16)
        plt.ylabel('N',fontsize=16)
        plt.title('2011-2013',fontsize=16)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        plt.legend(('Male','Female'),loc='upper left',fontsize='large',frameon=False)

        #print 'Sample size (Male,Female): ',wmale.sum(),wfemale.sum()#76,38, 33+-5% female
        print ' '
        print 'Time to faculty, Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmale]-phd_year[wmale]),np.mean(hire_year[wmale]-phd_year[wmale])/np.sqrt(wmale.sum()))
        print 'Time to faculty, Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfemale]-phd_year[wfemale]),np.mean(hire_year[wfemale]-phd_year[wfemale])/np.sqrt(wfemale.sum()))
        d,p = ks_2samp(hire_year[wmale]-phd_year[wmale],hire_year[wfemale]-phd_year[wfemale])
        print 'Prob: {:0.3f}'.format(p)
        
        wmale = (phd_year>2000) & (hire_year>=2014) & (hire_year<=2017) & (gender==0)
        wfemale = (phd_year>2000) & (hire_year>=2014) & (hire_year<=2017) & (gender==1)
        plt.subplot(122)
        p=plt.hist(hire_year[wmale]-phd_year[wmale],bins,color='k',lw=2,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0.,elinewidth=2)
        plt.axvline(np.mean(hire_year[wmale]-phd_year[wmale]),color='k',lw=2,ls=':')
        p=plt.hist(hire_year[wfemale]-phd_year[wfemale],bins,color='r',lw=2,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='or',capthick=0.,elinewidth=2)
        plt.axvline(np.mean(hire_year[wfemale]-phd_year[wfemale]),color='r',lw=2,ls=':')
        plt.xlabel('Time to hiring (years)',fontsize=16)
        plt.ylabel('N',fontsize=16)
        plt.title('2014-2017',fontsize=16)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        plt.legend(('Male','Female'),loc='upper left',fontsize='large',frameon=False)

        #print 'Sample size (Male,Female): ',wmale.sum(),wfemale.sum()#81,50, 38+-5% female
        print ' '
        print 'Time to faculty, Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmale]-phd_year[wmale]),np.mean(hire_year[wmale]-phd_year[wmale])/np.sqrt(wmale.sum()))
        print 'Time to faculty, Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfemale]-phd_year[wfemale]),np.mean(hire_year[wfemale]-phd_year[wfemale])/np.sqrt(wfemale.sum()))
        d,p = ks_2samp(hire_year[wmale]-phd_year[wmale],hire_year[wfemale]-phd_year[wfemale])
        print 'Prob: {:0.3f}'.format(p)

        # No substantial change in the hiring time distribution over the 7 years covered in this study. 
        
def plot_cc(file='misc.csv',full=True):
    '''Plot the hiring distribution, separated based on carnegire classification (R1 vs non-R1)'''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    gender,phd_year,hire_year,carnegie_class = load_demo(file)       
    wmale = (gender == 0) & (phd_year > 2000)
    wfemale = (gender == 1) & (phd_year > 2000)
    wr1 = (carnegie_class == 0) & (phd_year > 2000)
    wnr1 = (carnegie_class == 1) & (phd_year > 2000)
    wmr1 = (gender == 0) & (carnegie_class == 0) & (phd_year > 2000)
    wfr1 = (gender == 1) & (carnegie_class == 0) & (phd_year > 2000)
    wmnr1 = (gender == 0) & (carnegie_class == 1) & (phd_year > 2000)
    wfnr1 = (gender == 1) & (carnegie_class ==1) & (phd_year > 2000)

    #print 'R1: ',wmr1.sum(),wfr1.sum() #124, 58
    #print 'Not R1: ',wmnr1.sum(),wfnr1.sum() #33, 30

    bins = np.arange(0,11)-.5
    if full:
        #plot R1 vs non-R1, regardless of gender
        plt.rc('axes',lw=2)
        p=plt.hist(hire_year[wr1]-phd_year[wr1],bins,color='k',lw=2,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0,elinewidth=2)
        plt.axvline(np.mean(hire_year[wr1]-phd_year[wr1]),color='k',lw=2,ls=':')

        p=plt.hist(hire_year[wnr1]-phd_year[wnr1],bins,color='r',lw=2,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='or',capthick=0,elinewidth=2)
        plt.axvline(np.mean(hire_year[wnr1]-phd_year[wnr1]),color='r',lw=2,ls=':')
        plt.ylabel('N',fontsize=20)
        plt.xlabel('Time to hiring (years)',fontsize=20)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
        plt.legend(('R1','non-R1'),loc='upper left',fontsize=14,frameon=False)

        print ' '
        print 'Time to faculty, R1: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wr1]-phd_year[wr1]),np.mean(hire_year[wr1]-phd_year[wr1])/np.sqrt(wr1.sum()))
        print 'Time to faculty, non-R1: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wnr1]-phd_year[wnr1]),np.mean(hire_year[wnr1]-phd_year[wnr1])/np.sqrt(wnr1.sum()))
        d,p = ks_2samp(hire_year[wr1]-phd_year[wr1],hire_year[wnr1]-phd_year[wnr1])
        print 'Prob: {:0.3f}'.format(p)
        #The R1 vs non-R1 distributions are indistinguishable

    else:
        #Split apart male R1, female R1, male non-R1, femal non-R1

        plt.rc('axes',lw=3)
        plt.subplot(211)
        p=plt.hist(hire_year[wmr1]-phd_year[wmr1],bins,color='k',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0,elinewidth=3)
        plt.axvline(np.mean(hire_year[wmr1]-phd_year[wmr1]),color='k',lw=3,ls=':')

        p=plt.hist(hire_year[wfr1]-phd_year[wfr1],bins,color='r',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='or',capthick=0,elinewidth=3)
        plt.axvline(np.mean(hire_year[wfr1]-phd_year[wfr1]),color='r',lw=3,ls=':')
        plt.ylabel('N',fontweight='bold',fontsize=18)
        plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        plt.legend(('male R1','female R1'),loc='upper left',fontsize='large',frameon=False)

        print ' '
        print 'Time to faculty, male R1: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmr1]-phd_year[wmr1]),np.mean(hire_year[wmr1]-phd_year[wmr1])/np.sqrt(wmr1.sum()))
        print 'Time to faculty, female R1: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfr1]-phd_year[wfr1]),np.mean(hire_year[wfr1]-phd_year[wfr1])/np.sqrt(wfr1.sum()))
        d,p = ks_2samp(hire_year[wmr1]-phd_year[wmr1],hire_year[wfr1]-phd_year[wfr1])
        print 'Prob: {:0.3f}'.format(p)
        

        plt.subplot(212)
        p=plt.hist(hire_year[wmnr1]-phd_year[wmnr1],bins,color='k',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='ok',capthick=0,elinewidth=3)
        plt.axvline(np.mean(hire_year[wmnr1]-phd_year[wmnr1]),color='k',lw=3,ls=':')

        p=plt.hist(hire_year[wfnr1]-phd_year[wfnr1],bins,color='r',lw=3,histtype='step')
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]),fmt='or',capthick=0,elinewidth=3)
        plt.axvline(np.mean(hire_year[wfnr1]-phd_year[wfnr1]),color='r',lw=3,ls=':')
        plt.ylabel('N',fontweight='bold',fontsize=18)
        plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        plt.legend(('male non-R1','female non-R1'),loc='upper left',fontsize='large',frameon=False)

        print ' '
        print 'Time to faculty, male non-R1: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmnr1]-phd_year[wmnr1]),np.mean(hire_year[wmnr1]-phd_year[wmnr1])/np.sqrt(wmnr1.sum()))
        print 'Time to faculty, female non-R1: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfnr1]-phd_year[wfnr1]),np.mean(hire_year[wfnr1]-phd_year[wfnr1])/np.sqrt(wfnr1.sum()))
        d,p = ks_2samp(hire_year[wmnr1]-phd_year[wmnr1],hire_year[wfnr1]-phd_year[wfnr1])
        print 'Prob: {:0.3f}'.format(p)     
    
def basic_model(success=[1,1,1,1,1,1,1,1,1,1],plot_model=True,male_only=False,female_only=False):
    ''' Basic hiring model to fit overall distribution.'''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    if male_only:
        wmale = gender==0
        gender = gender[wmale]
        phd_year = phd_year[wmale]
        hire_year = hire_year[wmale]
    if female_only:
        wfemale = gender==1
        gender = gender[wfemale]
        phd_year = phd_year[wfemale]
        hire_year = hire_year[wfemale]

    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []

    mfrac = .7
    ffrac = 1-mfrac

    Nadd = 50000 #number of people added to the labor pool, per year

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(int(mfrac*Nadd)*[0,])
        labor_pool.extend(int(ffrac*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac*Nadd)+int(ffrac*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)
    
    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 30000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob/=prob.sum()
                
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])

        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(labor_pool))
        labor_pool = np.delete(labor_pool,index[w][select])
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])
        

    hired_pool = np.array(hired_pool)
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)

    if plot_model:
    #Plot full distribution
        bins=np.arange(0,11)-.5
        wuse = phd_year>2001
        plt.rc('axes',lw=3)
        p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
        plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3)
        plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',ls='--',lw=3)
        plt.ylabel('f',fontweight='bold',fontsize=24)
        plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=24)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
    
        
        print ' '
        print 'Mean time to hiring: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wuse]-phd_year[wuse]),np.mean(hire_year[wuse]-phd_year[wuse])/np.sqrt(wuse.sum()))

        wmodel = (phdy_hired_pool>2001) & (hirey_hired_pool > 2010)
        p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
        plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',ls='--',lw=3)
    
        print ' '
        print 'Mean time to hiring (model): {:0.2f}'.format(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]))

        #What fraction of astronomers that got a PhD in 2003 were hired?
        #print (phdy_hired_pool==2003).sum() 29932, which is ~60% of all people that got a PhD in 2003. This is much higher than the 15% that is measured by Hughes et al. ...
        

    else:
        bins=np.arange(0,11)-.5
        wuse = phd_year>2001
        hist,b = np.histogram(hire_year[wuse]-phd_year[wuse],bins)
        wmodel = (phdy_hired_pool>2001) & (hirey_hired_pool>2010)
        hist_model,b = np.histogram(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins)
        weight = (hist)
        hist_model = hist_model/float(hist_model.sum())
        hist = hist/float(hist.sum())
        return -((hist_model-hist)**2*weight).sum()


def fit_model():
    '''Fit a model to the data, to pull out the parameters of the model'''
    import numpy as np
    import emcee
    
    #Basic model, non-gendered
    gender,phd_year,hire_year,carnegie_class = load_demo()
    bins=np.arange(0,11)-.5
    wuse = phd_year>2001
    hist,b = np.histogram(hire_year[wuse]-phd_year[wuse],bins,density=True)
    ndim = 10
    nwalkers = 20
    p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))
    p0[:,0] = 0
    p0[:,5] = .5
    sampler = emcee.EnsembleSampler(nwalkers,ndim,basic_model,args=[False])
    pos,prob,state = sampler.run_mcmc(p0,1000)

    return sampler
#This doesn't converge, which is annoying...



#Semi-Infinite talent pool
#Basic model, full sample: success=[.02,.07,.25,.45,.57,.57,.35,.27,.20,.07]
#Basic model, men:         success=[.02,.05,.20,.38,.56,.60,.40,.32,.30,.12]
#Basic model, female:      success=[.00,.12,.35,.63,.66,.59,.31,.22,.08,.00]

#Realistic talent pool
#AIP (https://www.aip.org/sites/default/files/statistics/undergrad/enrolldegrees-a-12.3.pdf) reporst ~150 astronomy phds in 2012. According to the rumor mill, there are ~60 faculty hirings per year.
#Lets use Nadd=150 and Nhire=50 for a finite sample test
#Basic model, full sample, result from semi-infinite sample still works well (the increased scatter from the smaller sample makes it harder to estimate what parameters what and what doesn't...)

    

def plot_female_increase(slope=.1,year0=2000):
    '''Plot the change in the fraction of women with time according to the logistic model used in gendered_model1. Slope specifies the sharpness in the increase in women with time, while time0 is the year in which we reach gender parity among graduate students. '''
    import numpy as np
    import matplotlib.pyplot as plt

    #slope=.03,year0=2035
    
    years = np.arange(1960,2030)
    ffrac = 1/(1+np.exp(-slope*(years-year0)))
    mfrac = 1-ffrac
    labor_pool = []
    phdy_labor_pool = []
    Nadd = 10000
    
    for year in years:
        labor_pool.extend(int(mfrac[year-years[0]]*Nadd)*[0,])
        labor_pool.extend(int(ffrac[year-years[0]]*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac[year-years[0]]*Nadd)+int(ffrac[year-years[0]]*Nadd))*[year,])
    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #-- Grad student demographics --
    ffrac_grad = []
    for year in years:
        wmale_grad = (phdy_labor_pool>=(year)) & (phdy_labor_pool<=(year+7)) & (labor_pool==0)
        wfemale_grad = (phdy_labor_pool>=(year)) & (phdy_labor_pool<=(year+7)) & (labor_pool==1)
        ffrac_grad.append(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))
    ffrac_grad = np.array(ffrac_grad)
    
    plt.rc('axes',lw=2)
    plt.plot(years,1-ffrac_grad,color='k',lw=2)
    plt.plot(years,ffrac_grad,color='r',lw=2,ls='--')
    plt.legend(('Male','Female'),frameon=False,fontsize=16,loc='best')
    plt.xlabel('PhD Year',fontsize=20)
    plt.ylabel('Fraction (male,female)',fontsize=20)
    plt.errorbar([1992,1999,2003,2013],[.22,.26,.30,.34],yerr=[.02,.02,.02,.02],fmt='or')
    plt.xlim(1980,2020)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)

        
def gendered_model1(success=[1,1,1,1,1,1,1,1,1,1],slope=.01,year0=2000):
    '''A test of a gendered model.
       This model includes an increase in the fraction of women with time.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = [] #gender of each person in the labor pool. 0=male, 1=femal
    phdy_labor_pool = [] #PhD completion year of each person in the labor pool

    hired_pool = [] #gender of each person hired into a tenure track faculty position. 0=male, 1=female
    phdy_hired_pool = [] #PhD completion year of each hired astronomer
    hirey_hired_pool = [] #Year that each hired astronomer was hired
    ffrac_labor_pool = [] #Fraction of women among the labor pool in each year

    
    #mfrac = .74-slope*np.arange(17)
    #Use a logistic function for the fraction of women. Normalized so that the graduating class is 74% male in 2000
    years = np.arange(1980,2018)
    ffrac = 1/(1+np.exp(-slope*(years-year0)))
    mfrac = 1-ffrac

    Nadd = 30000 #number of people added to the labor pool, per year

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(int(mfrac[year-1980]*Nadd)*[0,])
        labor_pool.extend(int(ffrac[year-1980]*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac[year-1980]*Nadd)+int(ffrac[year-1980]*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Calculate the fraction of women within the labor market
        ffrac_labor_pool.append(float(labor_pool[w].sum())/w.sum())
        
        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob/=prob.sum()
                
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])

        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(labor_pool))
        labor_pool = np.delete(labor_pool,index[w][select])
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])

    hired_pool = np.array(hired_pool)
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    ffrac_labor_pool = np.array(ffrac_labor_pool)
    print '%Female labor market 2014 (19pm3%): {:0.2f}'.format(ffrac_labor_pool[-4])
    
    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))

    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=2)
    plt.subplot(211)
    plt.title('Male',fontsize=16)
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=2,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=2,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=2,label='Data')
    plt.ylabel('f',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=2,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=2,label='Model')
    plt.legend(loc='upper left',frameon=False,fontsize=14)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontsize=16)
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=2,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=2)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=2)
    plt.ylabel('f',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=2,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=2)
    
    



def gendered_model2(success=[1,1,1,1,1,1,1,1,1,1],bias=1.,Nhire=10000):
    '''A test of a gendered model.
       This model includes a bias towards hiring women. Here the bias is an increase if the relative probability of hiring a women compared to hiring a man
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []
    ffrac_labor_pool = []
    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd = 30000 #number of people added to the labor pool, per year

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(int(mfrac*Nadd)*[0,])
        labor_pool.extend(int(ffrac*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac*Nadd)+int(ffrac*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    #Nhire = 10000 #number of peopled hired per year
    if Nhire>Nadd:
        print 'Hiring more professors than PhDs being produced. Setting Nhire=.3*Nadd'
        Nhire = np.round(.3*Nadd)
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Calculate the fraction of women within the labor market
        ffrac_labor_pool.append(float(labor_pool[w].sum())/w.sum())
        
        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        wfemale = (labor_pool[w]==1)
        prob[wfemale]*=bias
        #wmale = (labor_pool[w]==0)
        #prob[wmale]/=bias
        prob/=prob.sum()
                
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])

        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(labor_pool))
        labor_pool = np.delete(labor_pool,index[w][select])
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])

    hired_pool = np.array(hired_pool)
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    ffrac_labor_pool = np.array(ffrac_labor_pool)
    print '%Female labor market 2014 (19pm3%): {:0.2f}'.format(ffrac_labor_pool[-4])

    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))
    
    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=2)
    plt.subplot(211)
    plt.title('Male',fontsize=16)
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=2,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=2,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=2,label='Data')
    plt.ylabel('f',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=2,histtype='step',normed=1,label=None)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=2,label='Model')
    plt.legend(loc='upper left',fontsize=14,frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontsize=16)
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=2,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=2)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=2)
    plt.ylabel('f',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=2,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=2)
#    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)


    #It is hard to find a bias that substantially changes the distribution for women... A bias of 100 (women are 100x more likely to be hired than men) marginally changes the distribution. Even then it has a tail towards long timescales that is higher than the data...
    

def gendered_model3(success=[1,1,1,1,1,1,1,1,1,1],tau_male=1.,tau_female=1.):
    '''A test of a gendered model.
       This model removes people from the labor pool at a rate that is proportional to the length of time they have been in the labor pool. The rate can vary between men and women
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []
    ffrac_labor_pool = []

    
    mfrac = .7
    ffrac = 1-mfrac

    year_start = 2 #when to start the exponential taper

    Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    Nadd_female = int(ffrac*30000)
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(Nadd_male*[0,])
        labor_pool.extend(Nadd_female*[1,])
        phdy_labor_pool.extend(Nadd_male*[year,])
        phdy_labor_pool.extend(Nadd_female*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))

                
    #Populate hired pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Calculate the fraction of women within the labor market
        ffrac_labor_pool.append(float(labor_pool[w].sum())/w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            if w_year.sum()>0:
                prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob/=prob.sum()
        
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])
               
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool))
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])
        labor_pool = np.delete(labor_pool,index[w][select])

        #Remove a fraction of non-hired candidates that have been in the labor pool for longer than year_start years
        taper_male = 1-np.exp(-(np.arange(15)-year_start)/float(tau_male))
        taper_male[taper_male<0] = 0 #fraction of non-hired candidates that are removed
        taper_female = 1-np.exp(-(np.arange(15)-year_start)/float(tau_female))
        taper_female[taper_female<0] = 0
        for i in range(year_start,10):
            index = np.arange(len(phdy_labor_pool))
            wmale = (phdy_labor_pool<=year) & (labor_pool==0)
            w_year = phdy_labor_pool[wmale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_male[i]),replace=False)
                
                phdy_labor_pool = np.delete(phdy_labor_pool,index[wmale][w_year][select])
                labor_pool = np.delete(labor_pool,index[wmale][w_year][select])
            index = np.arange(len(phdy_labor_pool))
            wfemale = (phdy_labor_pool<=year) & (labor_pool==1)
            w_year = phdy_labor_pool[wfemale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_female[i]),replace=False)
                phdy_labor_pool = np.delete(phdy_labor_pool,index[wfemale][w_year][select])
                labor_pool = np.delete(labor_pool,index[wfemale][w_year][select])
        
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    hired_pool = np.array(hired_pool)
    ffrac_labor_pool = np.array(ffrac_labor_pool)
    print '%Female labor market 2014 (19pm3%): {:0.2f}'.format(ffrac_labor_pool[-4])


    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))    
    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=2)
    plt.subplot(211)
    plt.title('Male',fontsize=16)
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=2,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=2,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=2,label='Data')
    plt.ylabel('f',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=2,histtype='step',normed=1,label=None)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=2,label='Model')
    plt.legend(loc='upper left',fontsize=14,frameon=False)
    
    #Female distribution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontsize=16)
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=2,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=2)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=2)
    plt.ylabel('f',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=2,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=2)
    
    #tau_male=4.,tau_female=3.,success=[.01,.02,.1,.2,.33,.36,.34,.48,1.,1.] does a reasonable job of fitting both distributions. In this case, women leave the labor pool one year faster than men.


def gendered_model3b(success=[1,1,1,1,1,1,1,1,1,1],tau_male=1.,tau_female=1.,slope=.03,year0=2035):
    '''A test of a gendered model.
       This model removes people from the labor pool at a rate that is proportional to the length of time they have been in the labor pool. The rate can vary between men and women. Unlike gendered_model3, this model includes an increase in the fraction of women getting PhDs with time (from gendered_model1). 
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []
    ffrac_labor_pool = []

    
    #mfrac = .7
    #ffrac = 1-mfrac
    years = np.arange(1980,2018)
    ffrac = 1/(1+np.exp(-slope*(years-year0)))
    mfrac = 1-ffrac

    year_start = 2 #when to start the exponential taper

    Nadd = 30000
    #Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    #Nadd_female = int(ffrac*30000)
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(int(mfrac[year-1980]*Nadd)*[0,])
        labor_pool.extend(int(ffrac[year-1980]*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac[year-1980]*Nadd)+int(ffrac[year-1980]*Nadd))*[year,])
        
        #labor_pool.extend(Nadd_male*[0,])
        #labor_pool.extend(Nadd_female*[1,])
        #phdy_labor_pool.extend(Nadd_male*[year,])
        #phdy_labor_pool.extend(Nadd_female*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))

               
    #Populate hired pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Calculate the fraction of women within the labor market
        ffrac_labor_pool.append(float(labor_pool[w].sum())/w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            if w_year.sum()>0:
                prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob/=prob.sum()
        
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])
               
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool))
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])
        labor_pool = np.delete(labor_pool,index[w][select])

        #Remove a fraction of non-hired candidates that have been in the labor pool for longer than year_start years
        taper_male = 1-np.exp(-(np.arange(15)-year_start)/float(tau_male))
        taper_male[taper_male<0] = 0 #fraction of non-hired candidates that are removed
        taper_female = 1-np.exp(-(np.arange(15)-year_start)/float(tau_female))
        taper_female[taper_female<0] = 0
        for i in range(year_start,10):
            index = np.arange(len(phdy_labor_pool))
            wmale = (phdy_labor_pool<=year) & (labor_pool==0)
            w_year = phdy_labor_pool[wmale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_male[i]),replace=False)
                
                phdy_labor_pool = np.delete(phdy_labor_pool,index[wmale][w_year][select])
                labor_pool = np.delete(labor_pool,index[wmale][w_year][select])
            index = np.arange(len(phdy_labor_pool))
            wfemale = (phdy_labor_pool<=year) & (labor_pool==1)
            w_year = phdy_labor_pool[wfemale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_female[i]),replace=False)
                phdy_labor_pool = np.delete(phdy_labor_pool,index[wfemale][w_year][select])
                labor_pool = np.delete(labor_pool,index[wfemale][w_year][select])
        
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    hired_pool = np.array(hired_pool)
    ffrac_labor_pool = np.array(ffrac_labor_pool)
    print '%Female labor market 2014 (19pm3%): {:0.2f}'.format(ffrac_labor_pool[-4])


    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))    
    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(211)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3,label='Data')
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1,label=None)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3,label='Model')
    plt.legend(loc='upper left',fontsize='large',frameon=False)
    
    #Female distribution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    
    #tau_male=4.,tau_female=3.,success=[.01,.02,.1,.2,.33,.36,.34,.48,1.,1.] does a reasonable job of fitting both distributions. In this case, women leave the labor pool one year faster than men.

    
def gendered_model3c(success=[1,1,1,1,1,1,1,1,1,1],tau_male=1.,tau_female=1.,tau=270.):
    '''A test of a gendered model.
       This model removes people from the labor pool at a rate that is proportional to the length of time they have been in the labor pool. The rate can vary between men and women. Unlike gendered_model3, this model includes an increase in the fraction of women getting PhDs with time (from gendered_model1). Unlike gendered_model3b, this model is extended over many years to consider the transition from assistant -> associate -> full professor.  
    '''


    #22% of grad students in 1992 are women
    #17% of assistant professors in 1992 are women
    #10% of associate professors in 1992 are women
    #5% of full professors in 1992 are women

    #26% of grad students in 1999 are women
    #18% of assistant professors in 1999 are women
    #14% of associate professors in 1999 are women
    #7% of full professors in 1999 are women

    #30% of grad students in 2003 are women
    #20% of assistant professors in 2003 are women
    #21% of associate professors in 2003 are women
    #9% of full professors in 2003 are women

    #34% of grad students in 2013 are women
    #26% of assistant professors in 2013 are women
    #19% of associate professors in 2013 are women
    #14% of associate professors in 2013 are women
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    year_begin,year_end = 1980,2050
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []
    ffrac_labor_pool = []

    
    years = np.arange(year_begin-40,year_end+10)
    #logistic model
    #ffrac = 1/(1+np.exp(-slope*(years-year0)))
    #mfrac = 1-ffrac

    #power law model
    ffrac = tau*((years-years[0])/float(years[-1]))**2.

    w = ffrac>1
    if w.sum()>0:
        ffrac[w] = 1
    mfrac = 1-ffrac

    year_start = 2 #when to start the exponential taper

    Nadd = 30000
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(year_begin-40,year_end+10):
        labor_pool.extend(int(mfrac[year-(year_begin-40)]*Nadd)*[0,])
        labor_pool.extend(int(ffrac[year-(year_begin-40)]*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac[year-(year_begin-40)]*Nadd)+int(ffrac[year-(year_begin-40)]*Nadd))*[year,])
        

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #-- Grad student demographics --
    ffrac_grad = []
    for year in range(year_begin,year_end):
        wmale_grad = (phdy_labor_pool>=(year)) & (phdy_labor_pool<=(year+7)) & (labor_pool==0)
        wfemale_grad = (phdy_labor_pool>=(year)) & (phdy_labor_pool<=(year+7)) & (labor_pool==1)
        ffrac_grad.append(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))
    
    #1992
    print '%Female 1992 grad students (22%): {:0.2f}'.format(ffrac_grad[1992-year_begin])

    #1999
    print '%Female 2003 grad students (26%): {:0.2f}'.format(ffrac_grad[1999-year_begin])

    #2003
    print '%Female 2003 grad students (30%): {:0.2f}'.format(ffrac_grad[2003-year_begin])

    #2013
    print '%Female 2003 grad students (34%): {:0.2f}'.format(ffrac_grad[2013-year_begin])

    #2040
    print '%Female 2040 grad students (??%): {:0.2f}'.format(ffrac_grad[2040-year_begin])

    #Ivie & Ray statistics:
    #Among astronomers that earned degrees between 1969 and 1981, 7% were female
    wmale = (phdy_labor_pool>=1969) & (phdy_labor_pool<=1981) & (labor_pool==0)
    wfemale = (phdy_labor_pool>=1969) & (phdy_labor_pool<=1981) & (labor_pool==1)
    print '%Female 1969-1981 grad students (7%): {:0.2f}'.format(float(wfemale.sum())/(wmale.sum()+wfemale.sum()))

    #13% for Phds between 1982 and 1990
    wmale = (phdy_labor_pool>=1982) & (phdy_labor_pool<=1990) & (labor_pool==0)
    wfemale = (phdy_labor_pool>=1982) & (phdy_labor_pool<=1990) & (labor_pool==1)
    print '%Female 1982-1990 grad students (13%): {:0.2f}'.format(float(wfemale.sum())/(wmale.sum()+wfemale.sum()))

    #19% for PhDs between 1993 and 1998
    wmale = (phdy_labor_pool>=1993) & (phdy_labor_pool<=1998) & (labor_pool==0)
    wfemale = (phdy_labor_pool>=1993) & (phdy_labor_pool<=1998) & (labor_pool==1)
    print '%Female 1993-1998 grad students (19%): {:0.2f}'.format(float(wfemale.sum())/(wmale.sum()+wfemale.sum()))
    
    #Populate hired pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(year_begin-30,year_end+10):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Calculate the fraction of women within the labor market
        ffrac_labor_pool.append(float(labor_pool[w].sum())/w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            if w_year.sum()>0:
                prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob/=prob.sum()
        
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])
               
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool))
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])
        labor_pool = np.delete(labor_pool,index[w][select])

        #Remove a fraction of non-hired candidates that have been in the labor pool for longer than year_start years
        taper_male = 1-np.exp(-(np.arange(15)-year_start)/float(tau_male))
        taper_male[taper_male<0] = 0 #fraction of non-hired candidates that are removed
        taper_female = 1-np.exp(-(np.arange(15)-year_start)/float(tau_female))
        taper_female[taper_female<0] = 0
        for i in range(year_start,10):
            index = np.arange(len(phdy_labor_pool))
            wmale = (phdy_labor_pool<=year) & (labor_pool==0)
            w_year = phdy_labor_pool[wmale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_male[i]),replace=False)
                
                phdy_labor_pool = np.delete(phdy_labor_pool,index[wmale][w_year][select])
                labor_pool = np.delete(labor_pool,index[wmale][w_year][select])
            index = np.arange(len(phdy_labor_pool))
            wfemale = (phdy_labor_pool<=year) & (labor_pool==1)
            w_year = phdy_labor_pool[wfemale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_female[i]),replace=False)
                phdy_labor_pool = np.delete(phdy_labor_pool,index[wfemale][w_year][select])
                labor_pool = np.delete(labor_pool,index[wfemale][w_year][select])
        
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    hired_pool = np.array(hired_pool)
    ffrac_labor_pool = np.array(ffrac_labor_pool)
    #print '\n%Female labor market 2014 (19pm3%): {:0.2f}'.format(ffrac_labor_pool[-4])


    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013

    #-- Assistant Professor demographics --
    ffrac_assistant = []
    for year in range(year_begin,year_end):
        wmale_prof = (hirey_hired_pool>=(year-6)) & (hirey_hired_pool<=year) & (hired_pool==0)
        wfemale_prof = (hirey_hired_pool>=(year-6)) & (hirey_hired_pool<=year) & (hired_pool==1)
        ffrac_assistant.append(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))
    
    #1992
    print '\n%Female Assistant Prof 1992 (17%): {:0.2f}'.format(ffrac_assistant[1992-year_begin])

    #1999
    print '%Female Assistant Prof 1999 (18%): {:0.2f}'.format(ffrac_assistant[1999-year_begin])

    #2003
    print '%Female Assistant Prof 2003 (20%): {:0.2f}'.format(ffrac_assistant[2003-year_begin])
    
    #2013
    print '%Female Assistant Prof 2013 (26%): {:0.2f}'.format(ffrac_assistant[2013-year_begin])

    #2040
    print '%Female Assistant Prof 2040 (??%): {:0.2f}'.format(ffrac_assistant[2040-year_begin])
   

    #-- Associate Professor demographics --
    #Assume associate professors occur from year 6 to year 15 of a faculty career
    ffrac_associate = []
    for year in range(year_begin,year_end):
        wmale_prof = (hirey_hired_pool>=(year-15)) & (hirey_hired_pool<=(year-6)) & (hired_pool==0)
        wfemale_prof = (hirey_hired_pool>=(year-15)) & (hirey_hired_pool<=(year-6)) & (hired_pool==1)
        ffrac_associate.append(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))
    
    #1992
    print '\n%Female Associate Prof 1992 (10%): {:0.2f}'.format(ffrac_associate[1992-year_begin])

    #1999
    print '%Female Associate Prof 1999 (14%): {:0.2f}'.format(ffrac_associate[1999-year_begin])

    #2003
    print '%Female Associate Prof 2003 (21%): {:0.2f}'.format(ffrac_associate[2003-year_begin])
    
    #2013
    print '%Female Associate Prof 2013 (19%): {:0.2f}'.format(ffrac_associate[2013-year_begin])

    #2040
    print '%Female Associate Prof 2040 (??%): {:0.2f}'.format(ffrac_associate[2040-year_begin])

    #-- Full Professor demographics --
    #Assume associate professors occur from year 15 to year 35 of a faculty career
    ffrac_full = []
    for year in range(year_begin,year_end):
        wmale_prof = (hirey_hired_pool>=(year-35)) & (hirey_hired_pool<=(year-15)) & (hired_pool==0)
        wfemale_prof = (hirey_hired_pool>=(year-35)) & (hirey_hired_pool<=(year-15)) & (hired_pool==1)
        ffrac_full.append(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))
    
    #1992
    print '\n%Female Full Prof 1992 (5%): {:0.2f}'.format(ffrac_full[1992-year_begin])

    #1999
    print '%Female Full Prof 1999 (7%): {:0.2f}'.format(ffrac_full[1999-year_begin])

    #2003
    print '%Female Full Prof 2003 (9%): {:0.2f}'.format(ffrac_full[2003-year_begin])
    
    #2013
    print '%Female Full Prof 2013 (14%): {:0.2f}'.format(ffrac_full[2013-year_begin])

    #2040
    print '%Female Full Prof 2040 (??%): {:0.2f}'.format(ffrac_full[2040-year_begin])



    plt.figure()
    year = np.arange(year_begin,year_end)
    plt.subplot(221)
    plt.plot(year,ffrac_grad,color='k',lw=2,ls='--')
    plt.plot([1992,1999,2003,2013],[.22,.26,.30,.34],'ok')
    plt.title('Grad Student')
    plt.ylabel('F$_{women}$',fontsize=14)
    plt.xlabel('Year',fontsize=14)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    plt.subplot(222)
    plt.plot(year,ffrac_assistant,color='r',lw=2,ls='--')
    plt.plot([1992,1999,2003,2013],[.17,.18,.20,.26],'or')
    plt.title('Assistant Professors')
    plt.ylabel('F$_{women}$',fontsize=14)
    plt.xlabel('Year',fontsize=14)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    plt.subplot(223)
    plt.plot(year,ffrac_associate,color='b',lw=2,ls='--')
    #plt.plot(year,.85*np.array(ffrac_associate),color='b',lw=2,ls=':')
    plt.plot([1992,1999,2003,2013],[.10,.14,.21,.19],'ob')
    plt.title('Associate Professors')
    plt.ylabel('F$_{women}$',fontsize=14)
    plt.xlabel('Year',fontsize=14)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    plt.subplot(224)
    plt.plot(year,ffrac_full,color='g',lw=2,ls='--')
    #plt.plot(year,.6*np.array(ffrac_full),color='g',lw=2,ls=':')
    plt.plot([1992,1999,2003,2013],[.05,.07,.09,.14],'og')
    plt.title('Full Professors')
    plt.ylabel('F$_{women}$',fontsize=14)
    plt.xlabel('Year',fontsize=14)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)


def gendered_model3d(success=[1,1,1,1,1,1,1,1,1,1],df=0.):
    '''A test of a gendered model.
       Similar to gendered_model3, this model removes people from the labor market. This model does it by removing a fixed fraction of women (df) per year. df=.1 removes 10% of female candidates, starting the fifth year on the labor market.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    phdy_labor_pool = []
    labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []
    

    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    Nadd_female = int(ffrac*30000)
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(Nadd_male*[0,])
        labor_pool.extend(Nadd_female*[1,])
        phdy_labor_pool.extend(Nadd_male*[year,])
        phdy_labor_pool.extend(Nadd_female*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w] == year-i
            if w_year.sum()>0:
                prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob/=prob.sum()

        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])
        
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool))
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])
        labor_pool = np.delete(labor_pool,index[w][select])


        #Removal a fraction of women that have been in the labor pool longer than five years
        wyear = (phdy_labor_pool<=(year-5)) & (labor_pool==1)
        index = np.arange(wyear.sum())
        if int(df*Nadd_female)>0:
            select = np.random.choice(index,int(df*Nadd_female),replace=False)
            phdy_labor_pool = np.delete(phdy_labor_pool,index[select])
            labor_pool = np.delete(labor_pool,index[select])

            
        
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    hired_pool = np.array(hired_pool)

    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))    

    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(211)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3,label='Data')
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hirey_hired_pool>2010) & (hired_pool==0)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1,label=None)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3,label='Model')
    plt.legend(loc='upper left',fontsize='large',frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hirey_hired_pool>2010) & (hired_pool==1)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    


def plot_df(tau_male=1.,tau_female=1.):
    '''Plot the taper on the labor market used in gendered_model3a. Used to show the relative rate at which men/women leave the labor market.'''

    import numpy as np
    import matplotlib.pyplot as plt

    year_start = 2
    
    taper_male = 1-np.exp(-(np.arange(15)-year_start)/float(tau_male))
    taper_male[taper_male<0] = 0
    taper_female = 1-np.exp(-(np.arange(15)-year_start)/float(tau_female))
    taper_female[taper_female<0] = 0

    plt.rc('axes',lw=3)
    plt.subplot(211)
    plt.plot(np.arange(15),taper_male,color='k',lw=3)
    plt.plot(np.arange(15),taper_female,color='r',lw=3)
    #plt.xlabel('Time since PhD (years)',fontweight='bold',fontsize=18)
    plt.ylabel('Fraction of PhDs \nleaving the labor market',fontweight='bold',fontsize=18)
    plt.legend(('Male','Female'),frameon=False,loc='upper left',fontsize='large')
    plt.xlim(0,14)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')

    plt.subplot(212)
    plt.plot(np.arange(15),taper_female/taper_male,color='k',lw=3)
    plt.ylabel('F(female)/F(male)',fontweight='bold',fontsize=18)
    plt.xlabel('Time since PhD (years)',fontweight='bold',fontsize=18)
    plt.xlim(0,14)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')


def gendered_model4(success=[1,1,1,1,1,1,1,1,1,1],fstar=.1,df=0.):
    '''A test of a gendered model.
    This model posits that a subset of the market consists of super-stars that are quickly hired into faculty positions. This leaves, at later times, a field dominated by non-stars. Female non-stars have a lower probability of being hired than male non-stars (due to e.g. unconcious bias). The turnover in the female distribution is due to the female superstars being quickly removed from the labor market, leaving behind the non-stars, who face a much tougher labor market than equivalent men. fstar=fraction of labor market that are super-stars. df=relative depression in hiring probability for women compared to men (df>0 means that non-star women are hired at a lower rate than men)'''

    import numpy as np
    import matplotlib.pyplot as plt

    gender,phd_year,hire_year,carnegie_class = load_demo()

    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf


    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []

    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    Nadd_female = int(ffrac*30000)

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(Nadd_male*[0,])
        labor_pool.extend(Nadd_female*[1,])
        phdy_labor_pool.extend(Nadd_male*[year,])
        phdy_labor_pool.extend(Nadd_female*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #Set a fraction of the female labor market to be super-stars
    index = (labor_pool==1)
    nstar = int(fstar*index.sum())
    select = np.random.choice(np.arange(index.sum()),nstar,replace=False)
    superstar = np.zeros(len(labor_pool))
    superstar[index][select] = 1

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))


    #Populate hired pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            if w_year.sum()>0:
                prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        prob[superstar[w]==1]*=100
        wfn = (labor_pool[w]==1) & (superstar[w]==0)
        prob[w][wfn]-=df
        prob[prob<0]=0.
        prob/=prob.sum()
        
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])
               
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool))
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])
        labor_pool = np.delete(labor_pool,index[w][select])
        superstar = np.delete(superstar,index[w][select])

    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    hired_pool = np.array(hired_pool)


    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))    
    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(211)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3,label='Data')
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1,label=None)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3,label='Model')
    plt.legend(loc='upper left',fontsize='large',frameon=False)
    
    #Female distribution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    
#This model doesn't seem to work... success=[.005,.01,.04,.1,.25,.56,.48,.37,.31,.08] with fstar=.15,df=1. at least fits the male distribution, but doesn't have much of an effect on the female distribution...


def gendered_model5(success=[1,1,1,1,1,1,1,1,1,1],tau=3.):
    '''A test of a gendered model.
       This model introduces a bias against hiring women, with the bias increasing as the time since a PhD increases. The probability is an exponential with a characteristic timescale given by tau.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    labor_pool = [] #gender of each person in the labor pool. 0=male, 1=femal
    phdy_labor_pool = [] #PhD completion year of each person in the labor pool

    hired_pool = [] #gender of each person hired into a tenure track faculty position. 0=male, 1=female
    phdy_hired_pool = [] #PhD completion year of each hired astronomer
    hirey_hired_pool = [] #Year that each hired astronomer was hired
    ffrac_labor_pool = [] #Fraction of women among the labor pool in each year

    
    #mfrac = .74-slope*np.arange(17)
    #Use a logistic function for the fraction of women. Normalized so that the graduating class is 74% male in 2000
    #years = np.arange(1980,2018)
    #ffrac = 1/(1+np.exp(-slope*(years-year0)))
    #mfrac = 1-ffrac
    mfrac = .7
    ffrac = 1-mfrac

        
    Nadd = 30000 #number of people added to the labor pool, per year

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1980,2018):
        labor_pool.extend(int(mfrac*Nadd)*[0,])
        labor_pool.extend(int(ffrac*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac*Nadd)+int(ffrac*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #2003 grad student demographics
    wmale_grad = (phdy_labor_pool >=2003) & (phdy_labor_pool <=2010) & (labor_pool==0)
    wfemale_grad = (phdy_labor_pool>=2003) & (phdy_labor_pool<=2010) & (labor_pool==1)
    print '%Female 2003 grad students (30pm2%): {:0.2f}'.format(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))


    
    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(1990,2018):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Calculate the fraction of women within the labor market
        ffrac_labor_pool.append(float(labor_pool[w].sum())/w.sum())
        
        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        wfemale = (labor_pool[w]==1)
        prob[wfemale]*=np.exp(-(year-phdy_labor_pool[w][wfemale])/tau)
        prob/=prob.sum()
                
        select = np.random.choice(index,Nhire,replace=False,p=prob)
        hired_pool.extend(labor_pool[w][select])
        phdy_hired_pool.extend(phdy_labor_pool[w][select])
        hirey_hired_pool.extend(Nhire*[year,])

        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(labor_pool))
        labor_pool = np.delete(labor_pool,index[w][select])
        phdy_labor_pool = np.delete(phdy_labor_pool,index[w][select])

    hired_pool = np.array(hired_pool)
    phdy_hired_pool=np.array(phdy_hired_pool)
    hirey_hired_pool=np.array(hirey_hired_pool)
    ffrac_labor_pool = np.array(ffrac_labor_pool)
    print '%Female labor market 2014 (19pm3%): {:0.2f}'.format(ffrac_labor_pool[-4])
    
    #Demographics to compare to Hughes et al. survey. This is the percentage of female grad students in 2003 (include anyone getting a phd in 2003-2010) and percentage of female assistant professors in 2013 (anyone hired from 2007-2013)
    #Hughes et al values: 30% grad students in 2003, 26% assistant professors in 2013
    wmale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==0)
    wfemale_prof = (hirey_hired_pool>=2007) & (hirey_hired_pool<=2013) & (hired_pool==1)
    print '%Female Assistant Prof 2013 (26pm4%): {:0.2f}'.format(float(wfemale_prof.sum())/(wmale_prof.sum()+wfemale_prof.sum()))

    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(211)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1,label=None)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3,label=None)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3,label='Data')
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3,label='Model')
    plt.legend(loc='upper left',fontsize='large',frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(212)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.errorbar(bins[:-1]+.5,p[0],yerr=np.sqrt(p[0]*wuse.sum())/wuse.sum(),fmt='ok',capthick=0.,elinewidth=3)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.ylabel('f',fontweight='bold',fontsize=18)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1) & (hirey_hired_pool>2010)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)


def plot_exp_demo(tau=10):
    '''Rather than a logistic function, use an exponential to describe the change in women with time (potentially a better match to the smaller fraction of women prior to 1990.'''

    import numpy as np
    import matplotlib.pyplot as plt
    
    #gender,phd_year,hire_year,carnegie_class = load_demo()
    year_begin,year_end = 1940,2060
    

    labor_pool = []
    phdy_labor_pool = []

    
    years = np.arange(year_begin,year_end)

    #logistic model
    #ffrac = 1/(1+np.exp(-slope*(years-year0)))
    #mfrac = 1-ffrac

    #exponential
    ffrac = np.exp((years-years[-1])/tau)


    #power law
    ffrac = tau*((years-years[0])/float(years[-1]))**2.

    w = ffrac>1
    if w.sum()>0:
        ffrac[w] = 1
    mfrac = 1-ffrac
    
    Nadd = 30000
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(year_begin,year_end):
        labor_pool.extend(int(mfrac[year-(year_begin)]*Nadd)*[0,])
        labor_pool.extend(int(ffrac[year-(year_begin)]*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac[year-(year_begin)]*Nadd)+int(ffrac[year-(year_begin)]*Nadd))*[year,])
        

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #-- Grad student demographics --
    ffrac_grad = []
    for year in range(year_begin,year_end):
        wmale_grad = (phdy_labor_pool>=(year)) & (phdy_labor_pool<=(year+7)) & (labor_pool==0)
        wfemale_grad = (phdy_labor_pool>=(year)) & (phdy_labor_pool<=(year+7)) & (labor_pool==1)
        ffrac_grad.append(float(wfemale_grad.sum())/(wmale_grad.sum()+wfemale_grad.sum()))
    
    #1992
    print '%Female 1992 grad students (22%): {:0.2f}'.format(ffrac_grad[1992-year_begin])

    #1999
    print '%Female 2003 grad students (26%): {:0.2f}'.format(ffrac_grad[1999-year_begin])

    #2003
    print '%Female 2003 grad students (30%): {:0.2f}'.format(ffrac_grad[2003-year_begin])

    #2013
    print '%Female 2003 grad students (34%): {:0.2f}'.format(ffrac_grad[2013-year_begin])

    #2040
    print '%Female 2040 grad students (??%): {:0.2f}'.format(ffrac_grad[2040-year_begin])

    #Ivie & Ray statistics:
    #Among astronomers that earned degrees between 1969 and 1981, 7% were female
    wmale = (phdy_labor_pool>=1969) & (phdy_labor_pool<=1981) & (labor_pool==0)
    wfemale = (phdy_labor_pool>=1969) & (phdy_labor_pool<=1981) & (labor_pool==1)
    print '%Female 1969-1981 grad students (7%): {:0.2f}'.format(float(wfemale.sum())/(wmale.sum()+wfemale.sum()))

    #13% for Phds between 1982 and 1990
    wmale = (phdy_labor_pool>=1982) & (phdy_labor_pool<=1990) & (labor_pool==0)
    wfemale = (phdy_labor_pool>=1982) & (phdy_labor_pool<=1990) & (labor_pool==1)
    print '%Female 1982-1990 grad students (13%): {:0.2f}'.format(float(wfemale.sum())/(wmale.sum()+wfemale.sum()))

    #19% for PhDs between 1993 and 1998
    wmale = (phdy_labor_pool>=1993) & (phdy_labor_pool<=1998) & (labor_pool==0)
    wfemale = (phdy_labor_pool>=1993) & (phdy_labor_pool<=1998) & (labor_pool==1)
    print '%Female 1993-1998 grad students (19%): {:0.2f}'.format(float(wfemale.sum())/(wmale.sum()+wfemale.sum()))
    
    plt.figure()
    #year = np.arange(year_begin,year_end)
    plt.plot(years,ffrac_grad,color='k',lw=2,ls='--')
    plt.plot([1992,1999,2003,2013],[.22,.26,.30,.34],'or')
    plt.plot([1969,1981],[.07,.07],color='r')
    plt.plot([1982,1990],[.13,.13],color='r')
    plt.plot([1993,1998],[.20,.20],color='r')
    plt.ylabel('F$_{women}$',fontsize=14)
    plt.xlabel('Year',fontsize=14)
    plt.xlim(1950,2040)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)



def plot_nphds():
    '''Using statistics collected by AIP (https://www.aip.org/statistics/rosters/astronomy), plot the number of PhDs awarded in astronomy for each year.'''

    import matplotlib.pyplot as plt

    year = [1992,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016]
    n = [93,117,133,126,111,116,88,139,101,102,88,116,93,119,125,161,141,156,160,152,155,147,130,170]

    plt.plot(year,n,'ok',ls='-')
    plt.xlabel('Year',fontsize=14)
    plt.ylabel('N(PhDs)',fontsize=14)
    plt.xticks(rotation=45)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)
