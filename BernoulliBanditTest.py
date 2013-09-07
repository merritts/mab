import time

import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

from BernoulliBandit import BernoulliBandit
from util import regret


#MPL defaults
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['axes.titlesize'] = 'large'
matplotlib.rcParams['xtick.labelsize'] = 'medium'
matplotlib.rcParams['ytick.labelsize'] = 'medium'
matplotlib.rcParams['axes.labelsize'] = 'large'
matplotlib.rcParams['axes.titlesize'] = 'large'
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


    
arms = 3
regrets = []
np.random.seed()

#draw random biases for each of the arms on the true bandit
true_arms = np.random.random_sample(arms)
arm_choices = np.zeros(arms)

#initialize random variables the true bandit using random bias values
bandit = [scipy.stats.bernoulli(a) for a in true_arms]

#initialize our model
bb = BernoulliBandit(arms=arms)

#setup plotting
plt.ion()
plt.show()

x = np.linspace(0,1,100)
plt.subplots_adjust(hspace=0.5)

for t in xrange(100):
    #choose a bandit probablistically, based on what we know so far
    arm = bb.choose_arm()
    
    #record choice for regret measurement
    arm_choices[arm]+=1

    #get a reward from a single trial from the arm
    reward = bandit[arm].rvs()
    
    #estimate the regret for this trial
    regrets.append(regret(true_arms.argmax(), true_arms, arm_choices))
    
    #update the model
    bb.update(arm,reward)
    
    #plot the estimated posteriors of each arm
    plt.subplot(211)
    plt.xlabel('Posterior', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    for a in range(len(bb.arms)):
        y = scipy.stats.beta.pdf(x,bb.arms[a].alpha,bb.arms[a].beta)
        plt.plot(x,y,label='Arm '+str(a)+' :'+str(true_arms[a])[:4])
    plt.legend(loc=0)
    plt.subplot(212)
    plt.plot(regrets, 'k-', label='Regret')
    plt.xlabel('Trial, t', fontsize=20)
    plt.ylabel('Total regret, r', fontsize=20)
    plt.draw()
    if t == 0:
        time.sleep(5)
    else:
        time.sleep(0.05)
    if t != 99:
        plt.clf()

#Print out summary of arm pulls, estimated biases and actual biases
print 'arm biases ', true_arms
print 'arm allocations ', arm_choices
print 'best estimated arm ', bb.best_arm()
print 'best actual arm ', true_arms.argmax()
