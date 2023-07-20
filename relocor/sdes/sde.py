
class SDE:
    
    def __init__(
            self,
            dim,
            dim_noise,
            drift,
            sigma,
            X0=None,
            batch_drift=None,
            batch_sigma=None,
            name='SDE',
        ):
        self.dim, self.dim_noise = dim, dim_noise
        self.drift = drift
        self.sigma = sigma
        self.X0 = X0
        self.batch_drift, self.batch_sigma = batch_drift, batch_sigma
        self.name = name

