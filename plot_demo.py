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

    #print 'Men: ',wmale.sum() #137
    #print 'Women: ',wfemale.sum() #78
    #print 'R1: ',wmr1.sum(),wfr1.sum() #106, 52
    #print 'Not R1: ',wmnr1.sum(),wfnr1.sum() #31, 26

    bins=np.arange(0,11)-.5
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
        
        print ' '
        print 'Mean time to hiring: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wuse]-phd_year[wuse]),np.mean(hire_year[wuse]-phd_year[wuse])/np.sqrt(wuse.sum()))
    else:
        #Plot distribution according to gender
        plt.rc('axes',lw=3)
        p=plt.hist(hire_year[wmale]-phd_year[wmale],bins,color='k',lw=3,histtype='step')
        plt.axvline(np.mean(hire_year[wmale]-phd_year[wmale]),color='k',lw=3)
        p=plt.hist(hire_year[wfemale]-phd_year[wfemale],bins,color='r',lw=3,histtype='step')
        plt.axvline(np.mean(hire_year[wfemale]-phd_year[wfemale]),color='r',lw=3)
        plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(18)
            tick.label1.set_fontweight('bold')
        plt.legend(('Male','Female'),loc='upper left',fontsize='large',frameon=False)

        print ' '
        print 'Time to faculty, Men: {:0.2f}+-{:0.2f}'.format(np.mean(hire_year[wmale]-phd_year[wmale]),np.mean(hire_year[wmale]-phd_year[wmale])/np.sqrt(wmale.sum()))
        print 'Time to faculty, Female: {:0.2f}+-{:0.2f} '.format(np.mean(hire_year[wfemale]-phd_year[wfemale]),np.mean(hire_year[wfemale]-phd_year[wfemale])/np.sqrt(wfemale.sum()))
        d,p = ks_2samp(hire_year[wmale]-phd_year[wmale],hire_year[wfemale]-phd_year[wfemale])
        print 'Prob: ',p

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
    for year in range(2000,2017):
        labor_pool.extend(int(mfrac*Nadd)*[0,])
        labor_pool.extend(int(ffrac*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac*Nadd)+int(ffrac*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)
    

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 30000 #number of peopled hired per year
    for year in range(2011,2017):
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
        plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',ls='--',lw=3)
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

        wmodel = phdy_hired_pool>2001
        p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
        plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',ls='--',lw=3)
    
        print ' '
        print 'Mean time to hiring (model): {:0.2f}'.format(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]))

    else:
        bins=np.arange(0,11)-.5
        wuse = phd_year>2001
        hist,b = np.histogram(hire_year[wuse]-phd_year[wuse],bins)
        wmodel = phdy_hired_pool>2001
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

    
    
        
def gendered_model1(success=[1,1,1,1,1,1,1,1,1,1],slope=.01):
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

    labor_pool = []
    phdy_labor_pool = []

    hired_pool = []
    phdy_hired_pool = []
    hirey_hired_pool = []

    
    mfrac = .7-slope*np.arange(17)
    ffrac = 1-mfrac
    if (mfrac<0).sum()>0:
        print 'Fraction of men becomes negative. Choose a shallower slope.'
        return

    Nadd = 30000 #number of people added to the labor pool, per year

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(2000,2017):
        labor_pool.extend(int(mfrac[year-2000]*Nadd)*[0,])
        labor_pool.extend(int(ffrac[year-2000]*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac[year-2000]*Nadd)+int(ffrac[year-2000]*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(2011,2017):
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


    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(121)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(122)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)

    #If slope=.04 (basically no men today) then success=[.03,.07,.3,.5,.65,.60,.36,.25,.23,.08] can reasonably fit both distributions
    #If slope=.01 (nearly 50/50 today) then success=[.03,.06,.23,.43,.61,.65,.45,.32,.32,.12] gives a reasonable fit to the men, but does not do a good job of fitting the women
    

def gendered_model2(success=[1,1,1,1,1,1,1,1,1,1],bias=1):
    '''A test of a gendered model.
       This model includes bias towards hiring women (either because women are more qualified for faculty jobs, or because efforts at diversity in hiring lead to more frequenct hiring of women). The number of men hired per year is Nhire, while the number of women hired per year is bias*Nhire.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    gender,phd_year,hire_year,carnegie_class = load_demo()
    
    wnonzero = np.array(success)<0
    wlarge = np.array(success)>1
    if wnonzero.sum()>0 or wlarge.sum()>0:
        return -np.inf

    phdy_labor_pool_male = []
    phdy_labor_pool_female = []

    phdy_hired_pool_male = []
    hirey_hired_pool_male = []
    phdy_hired_pool_female = []
    hirey_hired_pool_female = []

    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    Nadd_female = int(ffrac*30000)
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(2000,2017):
        phdy_labor_pool_male.extend(Nadd_male*[year,])
        phdy_labor_pool_female.extend(Nadd_female*[year,])

    phdy_labor_pool_male = np.array(phdy_labor_pool_male)
    phdy_labor_pool_female = np.array(phdy_labor_pool_female)

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(2011,2017):
        wmale = phdy_labor_pool_male<=year
        index_male = np.arange(wmale.sum())
        wfemale = phdy_labor_pool_female<=year
        index_female = np.arange(wfemale.sum())

        #Set relative probablities of selecting different candidates
        prob_male = np.ones(wmale.sum())
        prob_female = np.ones(wfemale.sum())
        for i in range(10):
            w_year = phdy_labor_pool_male[wmale]==year-i
            prob_male[w_year]*=success[i]
            w_year = phdy_labor_pool_female[wfemale]==year-i
            prob_female[w_year]*=success[i]
        w_too_old = phdy_labor_pool_male[wmale]<=(year-10)
        prob_male[w_too_old] = 0
        prob_male/=prob_male.sum()
        w_too_old = phdy_labor_pool_female[wfemale]<=(year-10)
        prob_female[w_too_old] = 0
        prob_female/=prob_female.sum()
        
        select_male = np.random.choice(index_male,Nhire,replace=False,p=prob_male)
        select_female = np.random.choice(index_female,int(bias*Nhire),replace=False,p=prob_female)
        phdy_hired_pool_male.extend(phdy_labor_pool_male[wmale][select_male])
        hirey_hired_pool_male.extend(Nhire*[year,])
        phdy_hired_pool_female.extend(phdy_labor_pool_female[wfemale][select_female])
        hirey_hired_pool_female.extend(int(bias*Nhire)*[year,])
        
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool_male))
        phdy_labor_pool_male = np.delete(phdy_labor_pool_male,index[wmale][select_male])
        index = np.arange(len(phdy_labor_pool_female))
        phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[wfemale][select_female])
        
    #hired_pool = np.array(hired_pool)
    phdy_hired_pool_male=np.array(phdy_hired_pool_male)
    hirey_hired_pool_male=np.array(hirey_hired_pool_male)
    phdy_hired_pool_female=np.array(phdy_hired_pool_female)
    hirey_hired_pool_female=np.array(hirey_hired_pool_female)

    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(121)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool_male>2001)# & (hired_pool==0)
    p = plt.hist(hirey_hired_pool_male[wmodel]-phdy_hired_pool_male[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool_male[wmodel]-phdy_hired_pool_male[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(122)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool_female>2001)# & (hired_pool==1)
    p = plt.hist(hirey_hired_pool_female[wmodel]-phdy_hired_pool_female[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool_female[wmodel]-phdy_hired_pool_female[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)


    #bias=1.5, success=[.01,.05,.18,.35,.55,.59,.43,.33,.3,.1] sort of works. It fits the male distribution, and gets the right average for the female distribution. It doesn't match the full female distribution. It overestimates to fraction at 1-2 years, underestimates from 3-6 years, and overestimates at 8-9 years. Overall the model distribution is flatter than the observed distribution.
    #bias has to be <2.0 because there aren't enough women for a higher bias...
    #bias=1.2 fits the short-timescale (<=2 years) side of the distribution, but does much worse on the high timescale (3-9 years) side of the distribution


def gendered_model2b(success=[1,1,1,1,1,1,1,1,1,1],bias=1.):
    '''A test of a gendered model.
       This model includes a bias towards hiring women. Unlike gendered_model2, here the bias is an increase if the relative probability of hiring a women compared to hiring a man
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

    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd = 30000 #number of people added to the labor pool, per year

    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(2000,2017):
        labor_pool.extend(int(mfrac*Nadd)*[0,])
        labor_pool.extend(int(ffrac*Nadd)*[1,])
        phdy_labor_pool.extend((int(mfrac*Nadd)+int(ffrac*Nadd))*[year,])

    labor_pool = np.array(labor_pool)
    phdy_labor_pool = np.array(phdy_labor_pool)

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(2011,2017):
        w = phdy_labor_pool<=year
        index = np.arange(w.sum())

        #Set relative probablities of selecting different candidates
        prob = np.ones(w.sum())
        for i in range(10):
            w_year = phdy_labor_pool[w]==year-i
            prob[w_year]*=success[i]
        w_too_old = phdy_labor_pool[w]<=(year-10)
        prob[w_too_old] = 0
        wfemale = (labor_pool[w]==1)
        prob[wfemale]*=bias
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


    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(121)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==0)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(122)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool>2001) & (hired_pool==1)
    p = plt.hist(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool[wmodel]-phdy_hired_pool[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)


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

    phdy_labor_pool_male = []
    phdy_labor_pool_female = []

    phdy_hired_pool_male = []
    hirey_hired_pool_male = []
    phdy_hired_pool_female = []
    hirey_hired_pool_female = []

    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    Nadd_female = int(ffrac*30000)
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1995,2017):
        phdy_labor_pool_male.extend(Nadd_male*[year,])
        phdy_labor_pool_female.extend(Nadd_female*[year,])

    phdy_labor_pool_male = np.array(phdy_labor_pool_male)
    phdy_labor_pool_female = np.array(phdy_labor_pool_female)

    #Initial removal of women from the labor market (sets up distribution for year=2011)
    for year in range(2005,2011):
        taper_male = 1-np.exp(-(np.arange(10)-4)/float(tau_male))
        taper_male[taper_male<0]
        taper_female = 1-np.exp(-(np.arange(10)-4)/float(tau_female))
        taper_female[taper_female<0] = 0
        for i in range(5,10):
            index = np.arange(len(phdy_labor_pool_male))
            wmale = phdy_labor_pool_male<=year
            w_year = phdy_labor_pool_male[wmale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_male[i]),replace=False)
                phdy_labor_pool_male = np.delete(phdy_labor_pool_male,index[wmale][w_year][select])
            index = np.arange(len(phdy_labor_pool_female))
            wfemale = phdy_labor_pool_female<=year
            w_year = phdy_labor_pool_female[wfemale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_female[i]),replace=False)
                phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[wfemale][w_year][select])

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(2011,2017):
        wmale = phdy_labor_pool_male<=year
        index_male = np.arange(wmale.sum())
        wfemale = phdy_labor_pool_female<=year
        index_female = np.arange(wfemale.sum())

        #Set relative probablities of selecting different candidates
        prob_male = np.ones(wmale.sum())
        prob_female = np.ones(wfemale.sum())
        for i in range(10):
            w_year = phdy_labor_pool_male[wmale]==year-i
            if w_year.sum()>0:
                prob_male[w_year]*=success[i]
            w_year = phdy_labor_pool_female[wfemale]==year-i
            if w_year.sum()>0:
                prob_female[w_year]*=success[i]
        w_too_old = phdy_labor_pool_male[wmale]<=(year-10)
        prob_male[w_too_old] = 0
        prob_male/=prob_male.sum()
        w_too_old = phdy_labor_pool_female[wfemale]<=(year-10)
        prob_female[w_too_old] = 0
        prob_female/=prob_female.sum()
        
        select_male = np.random.choice(index_male,Nhire,replace=False,p=prob_male)
        select_female = np.random.choice(index_female,Nhire,replace=False,p=prob_female)
        phdy_hired_pool_male.extend(phdy_labor_pool_male[wmale][select_male])
        hirey_hired_pool_male.extend(Nhire*[year,])
        phdy_hired_pool_female.extend(phdy_labor_pool_female[wfemale][select_female])
        hirey_hired_pool_female.extend(Nhire*[year,])
        
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool_male))
        phdy_labor_pool_male = np.delete(phdy_labor_pool_male,index[wmale][select_male])
        index = np.arange(len(phdy_labor_pool_female))
        phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[wfemale][select_female])

        #Remove a fraction of non-hired candidates that have been in the labor pool for longer than 4 years
        taper_male = 1-np.exp(-(np.arange(15)-4)/float(tau_male))
        taper_male[taper_male<0] = 0 #fraction of non-hired candidates that are removed
        taper_female = 1-np.exp(-(np.arange(15)-4)/float(tau_female))
        taper_female[taper_female<0] = 0
        for i in range(5,10):
            index = np.arange(len(phdy_labor_pool_male))
            wmale = phdy_labor_pool_male<=year
            w_year = phdy_labor_pool_male[wmale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_male[i]),replace=False)
                phdy_labor_pool_male = np.delete(phdy_labor_pool_male,index[wmale][w_year][select])
            index = np.arange(len(phdy_labor_pool_female))
            wfemale = phdy_labor_pool_female<=year
            w_year = phdy_labor_pool_female[wfemale]==year-i
            if w_year.sum()>0:
                select = np.random.choice(np.arange(w_year.sum()),int(w_year.sum()*taper_female[i]),replace=False)
                phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[wfemale][w_year][select])
            
        
    phdy_hired_pool_male=np.array(phdy_hired_pool_male)
    hirey_hired_pool_male=np.array(hirey_hired_pool_male)
    phdy_hired_pool_female=np.array(phdy_hired_pool_female)
    hirey_hired_pool_female=np.array(hirey_hired_pool_female)

    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(121)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool_male>2001)
    p = plt.hist(hirey_hired_pool_male[wmodel]-phdy_hired_pool_male[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool_male[wmodel]-phdy_hired_pool_male[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)
    
    #Female distribution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(122)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool_female>2001)
    p = plt.hist(hirey_hired_pool_female[wmodel]-phdy_hired_pool_female[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool_female[wmodel]-phdy_hired_pool_female[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)

    #tau_male=4.,tau_female=3.,success=[.01,.02,.1,.2,.33,.36,.34,.48,1.,1.] does a reasonable job of fitting both distributions. In this case, women leave the labor pool one year faster than men.


def gendered_model3b(success=[1,1,1,1,1,1,1,1,1,1],df=0.):
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

    phdy_labor_pool_male = []
    phdy_labor_pool_female = []

    phdy_hired_pool_male = []
    hirey_hired_pool_male = []
    phdy_hired_pool_female = []
    hirey_hired_pool_female = []

    
    mfrac = .7
    ffrac = 1-mfrac

    Nadd_male = int(mfrac*30000) #number of people added to the labor pool, per year
    Nadd_female = int(ffrac*30000)
    
    #Populate labor pool. Assume the number of people added per year is constant, and the fraction of women is constant per year
    for year in range(1995,2017):
        phdy_labor_pool_male.extend(Nadd_male*[year,])
        phdy_labor_pool_female.extend(Nadd_female*[year,])

    phdy_labor_pool_male = np.array(phdy_labor_pool_male)
    phdy_labor_pool_female = np.array(phdy_labor_pool_female)

    #Initial removal of women from the labor market (sets up the distribution for year=2011).
    for year in range(2005,2011):
        wyear = phdy_labor_pool_female<=(year-5)
        index = np.arange(wyear.sum())
        if int(df*Nadd_female)>0:
            select = np.random.choice(index,int(df*Nadd_female),replace=False)
            phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[select])
        

    #Populate hired pool. Start 5 years after beginning of labor pool. Randomly select people from labor pool
    Nhire = 10000 #number of peopled hired per year
    for year in range(2011,2017):
        wmale = phdy_labor_pool_male<=year
        index_male = np.arange(wmale.sum())
        wfemale = phdy_labor_pool_female<=year
        index_female = np.arange(wfemale.sum())

        #Set relative probablities of selecting different candidates
        prob_male = np.ones(wmale.sum())
        prob_female = np.ones(wfemale.sum())
        for i in range(10):
            w_year = phdy_labor_pool_male[wmale]==year-i
            if w_year.sum()>0:
                prob_male[w_year]*=success[i]
            w_year = phdy_labor_pool_female[wfemale]==year-i
            if w_year.sum()>0:
                prob_female[w_year]*=success[i]
        w_too_old = phdy_labor_pool_male[wmale]<=(year-10)
        prob_male[w_too_old] = 0
        prob_male/=prob_male.sum()
        w_too_old = phdy_labor_pool_female[wfemale]<=(year-10)
        prob_female[w_too_old] = 0
        prob_female/=prob_female.sum()
        
        select_male = np.random.choice(index_male,Nhire,replace=False,p=prob_male)
        select_female = np.random.choice(index_female,Nhire,replace=False,p=prob_female)
        phdy_hired_pool_male.extend(phdy_labor_pool_male[wmale][select_male])
        hirey_hired_pool_male.extend(Nhire*[year,])
        phdy_hired_pool_female.extend(phdy_labor_pool_female[wfemale][select_female])
        hirey_hired_pool_female.extend(Nhire*[year,])
        
        #Remove hired candidates from the pool of available candidates
        index = np.arange(len(phdy_labor_pool_male))
        phdy_labor_pool_male = np.delete(phdy_labor_pool_male,index[wmale][select_male])
        index = np.arange(len(phdy_labor_pool_female))
        phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[wfemale][select_female])

        #Removal a fraction of women that have been in the labor pool longer than five years
        wyear = phdy_labor_pool_female<=(year-5)
        index = np.arange(wyear.sum())
        if int(df*Nadd_female)>0:
            select = np.random.choice(index,int(df*Nadd_female),replace=False)
            phdy_labor_pool_female = np.delete(phdy_labor_pool_female,index[select])

            
        
    phdy_hired_pool_male=np.array(phdy_hired_pool_male)
    hirey_hired_pool_male=np.array(hirey_hired_pool_male)
    phdy_hired_pool_female=np.array(phdy_hired_pool_female)
    hirey_hired_pool_female=np.array(hirey_hired_pool_female)

    #Plot full distribution
    bins=np.arange(0,11)-.5
    #Male distrobution
    wuse = (phd_year>2001) & (gender==0)
    plt.rc('axes',lw=3)
    plt.subplot(121)
    plt.title('Male',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool_male>2001)
    p = plt.hist(hirey_hired_pool_male[wmodel]-phdy_hired_pool_male[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool_male[wmodel]-phdy_hired_pool_male[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)
    
    #Female distrobution
    wuse = (phd_year>2001) & (gender==1)
    plt.subplot(122)
    plt.title('Female',fontweight='bold')
    p = plt.hist(hire_year[wuse]-phd_year[wuse],bins,color='k',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hire_year[wuse]-phd_year[wuse]),color='k',lw=3)
    plt.xlabel('Time to hiring (years)',fontweight='bold',fontsize=18)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')
    
        
    wmodel = (phdy_hired_pool_female>2001)
    p = plt.hist(hirey_hired_pool_female[wmodel]-phdy_hired_pool_female[wmodel],bins,color='r',lw=3,histtype='step',normed=1)
    plt.axvline(np.mean(hirey_hired_pool_female[wmodel]-phdy_hired_pool_female[wmodel]),color='r',lw=3)
    plt.legend(('Data','Model'),loc='upper left',fontsize='large',frameon=False)

    #success=[.02,.04,.18,.35,.55,.64,.44,.33,.31,.12] with df=.8 fits some of the distribution, but is way too high of a df value...
