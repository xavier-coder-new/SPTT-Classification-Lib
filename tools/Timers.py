class Timers:
    def __init__(self):
        self.krank = None
        self.batch_time = 0.0
        self.epoch_time = 0.0
        self.total_time = 0.0
        self.cal_gradient_time = 0.0

    def reset_time(self):
        self.batch_time = 0.0
        self.epoch_time = 0.0
        self.total_time = 0.0
        self.cal_gradient_time = 0.0
    
    # HACK: This is a workaround to allow the model to access and update the time counts without passing them as arguments.
    def update_batch_time(self, period: float):
        self.batch_time += period
        return self.batch_time
        
    def update_epoch_time(self, period: float):
        self.epoch_time += period
        return self.epoch_time
    
    def update_total_time(self, period: float):
        self.total_time += period
        return self.total_time
    
    def update_cal_gradient_time(self, period: float):
        self.cal_gradient_time = period
        return self.cal_gradient_time