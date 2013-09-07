import scipy.stats

class BetaBinomial(object):
    
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta
    
    def update(self, x):
        if x == 0.:
            self.beta+=1.
        elif x == 1.:
            self.alpha+=1.
    
    def get_mean(self):
        return self.alpha / (self.alpha + self.beta)
    
    def sample(self, samples):
        if samples > 1:
            return scipy.stats.beta.rvs(self.alpha, self.beta, size=samples).mean()
        else:
            return scipy.stats.beta.rvs(self.alpha, self.beta, size=samples)