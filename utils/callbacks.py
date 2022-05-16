import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import time

class ReportCallback(keras.callbacks.Callback):
    def __init__(self, q, sleep = None, save_on_epoch_end = True):
        super(ReportCallback, self).__init__()
        self.q = q
        self.sleep = sleep
        self.save_on_epoch_end = save_on_epoch_end
        
    def on_train_begin(self, logs=None):
        self.q.put({"time":time.time(), "event": "train_begin", "msg" : "Start training"})
        print(self.params)
        if self.sleep:
            self.sleep.sleep(0)

    def on_train_end(self, logs=None):
        self.q.put({"time":time.time(), "event": "train_end", "msg" : "Train ended", "matric" : logs})
        if self.sleep:
            self.sleep.sleep(0)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.q.put({"time":time.time(), "event": "epoch_begin", "msg" : "Start epoch", "epoch" : epoch})
        if self.sleep:
            self.sleep.sleep(0)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.save_on_epoch_end:
            self.q.put({"time":time.time(), "event": "epoch_end", "msg" : "End epoch", "epoch" : epoch, "matric" : logs})
            if self.sleep:
                self.sleep.sleep(0)
            
    def on_test_begin(self, logs=None):
        self.q.put({"time":time.time(), "event": "test_begin", "msg" : "Start testing"})
        if self.sleep:
            self.sleep.sleep(0)
        
    def on_test_end(self, logs=None):
        self.q.put({"time":time.time(), "event": "test_end", "msg" : "Test ended", "matric" : logs})
        if self.sleep:
            self.sleep.sleep(0)
        
    def on_predict_begin(self, logs=None):
        self.q.put({"time":time.time(), "event": "predict_begin", "msg" : "Start predicting"})
        if self.sleep:
            self.sleep.sleep(0)
        
    def on_predict_end(self, logs=None):
        self.q.put({"time":time.time(), "event": "predict_end", "msg" : "Predict ended", "matric" : logs})
        if self.sleep:
            self.sleep.sleep(0)

    def on_train_batch_begin(self, batch, logs=None):
        res = {"time":time.time(), "event": "train_batch_begin", "msg" : "Start train batch", "batch" : batch}
        if self.params and 'steps' in self.params:
            res["steps"] = self.params["steps"]
        self.q.put(res)
        if self.sleep:
            self.sleep.sleep(0)

    def on_train_batch_end(self, batch, logs=None):
        res = {"time":time.time(), "event": "train_batch_end", "msg" : "Train batch ended", "batch" : batch, "matric" : logs}
        if self.params and 'steps' in self.params:
            res["steps"] = self.params["steps"]
        self.q.put(res)
        if self.sleep:
            self.sleep.sleep(0)

    # def on_test_batch_begin(self, batch, logs=None):
    #     self.q.put({"time":time.time(), "event": "test_batch_begin", "msg" : "Start test batch", "batch" : batch})
    #     if self.sleep:
    #         self.sleep.sleep(0)

    # def on_test_batch_end(self, batch, logs=None):
    #     self.q.put({"time":time.time(), "event": "test_batch_end", "msg" : "Test batch ended", "batch" : batch, "matric" : logs})
    #     if self.sleep:
    #         self.sleep.sleep(0)

    # def on_predict_batch_begin(self, batch, logs=None):
    #     self.q.put({"time":time.time(), "event": "predict_batch_begin", "msg" : "Start predict batch", "batch" : batch})
    #     if self.sleep:
    #         self.sleep.sleep(0)

    # def on_predict_batch_end(self, batch, logs=None):
    #     self.q.put({"time":time.time(), "event": "predict_batch_end", "msg" : "Predict batch ended", "batch" : batch, "matric" : logs})
    #     if self.sleep: 
    #         self.sleep.sleep(0)
    
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []
        self.current_lr = 0.0
        
    def on_epoch_end(self, epoch, logs={}):
        if self.verbose == 1:
            print('Epoch %05d: Learning rate is %s.\n' % (epoch, self.current_lr))        

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        self.current_lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, self.current_lr)
        if self.verbose ==2:
            print('\nBatch %05d: setting learning rate to %s.' % (self.global_step + 1, self.current_lr))

