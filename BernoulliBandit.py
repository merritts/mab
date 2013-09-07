import numpy as np

from BetaBinomial import BetaBinomial

class BernoulliBandit(object):
    
    def __init__(self, arms=2, samples=1):
        self.arms = [BetaBinomial() for i in xrange(arms)]
        self.samples = samples
    
    def update(self, arm, reward):
        self.arms[arm].update(reward)
    
    def choose_arm(self):
        #randomized probability sampling: estimate mean by sampling from the posterior
        #generated by trials, choose arm with the largest mean
        return np.argmax([arm.sample(samples=self.samples) for arm in self.arms])
        
    def best_arm(self):
        return np.argmax([arm.get_mean() for arm in self.arms])
    
    def get_mean(self, arm):
        return self.arms[arm].get_mean()