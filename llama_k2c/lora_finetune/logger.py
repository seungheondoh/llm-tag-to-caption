import os
import torch
import time
from omegaconf import OmegaConf


class Logger:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.init_training_log()

    def init_training_log(self):
        self.log_filename = os.path.join(self.experiment_dir, "train_log.tsv")
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            log_file = open(self.log_filename, 'a')
            log_file.write(
                'step\ttrain_loss\ttime_stamp\tlearning_rate\n'
            )
            log_file.close()
            
    def get_timestamp(self):
        return str(time.strftime('%Y-%m-%d-%H_%M_%S', time.gmtime()))

    def write(self, text):
        print(text)

    def update_training_log(self, step, train_loss,lr):
        time_stamp = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        self.write(
            'Step %d, train loss %g, time-stamp %s, lr %s'
            %
            (step, train_loss, time_stamp, lr))

        log_file = open(self.log_filename, 'a')
        log_file.write('%d\t%g\t%s\t%s\n' %
                       (step, train_loss,
                        time_stamp,lr))
        log_file.close()