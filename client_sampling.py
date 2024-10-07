'''
Sunwoo Lee, Ph.D.
<sunwool@inha.ac.kr>
2023.03.05
'''
from mpi4py import MPI
import numpy as np
import math
import time
import tensorflow as tf

class sampling:
    def __init__ (self, num_clients, num_workers, num_candidates):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = num_workers
        self.num_clients = num_clients
        self.num_candidates = num_candidates
        self.num_local_workers = int(num_workers / self.size)
        self.num_local_clients = int(num_clients / self.size)
        self.local_losses = np.full((self.num_clients), np.Inf)
        self.fixed_losses = np.full((self.num_clients), np.Inf)
        self.local_norms = np.zeros((self.num_clients))
        self.avg_norms = np.zeros((self.num_clients))
        self.num_updates = np.zeros((self.num_clients))
        self.active_devices = np.zeros((self.num_clients))
        self.rng = np.random.default_rng(int(time.time()))
        np.random.seed(int(time.time()))

    def random (self):
        self.active_devices = np.random.choice(np.arange(self.num_clients), size = self.num_workers, replace = False)
        self.active_devices = self.comm.bcast(self.active_devices, root = 0)
        return self.active_devices

    def power_of_choice (self, epoch_id):
        # 1. Sample M clients randomly without replacements.
        candidates = self.rng.choice(np.arange(self.num_clients), size = self.num_candidates, replace = False)
        self.active_devices = candidates[np.argsort(self.local_losses[candidates])[-self.num_workers:]]
        self.active_devices = self.comm.bcast(self.active_devices, root = 0)
        return self.active_devices

    def power_of_choice2 (self, epoch_id):
        # 1. Sample M clients randomly without replacements.
        border = (self.num_clients - self.num_workers) // 4
        candidates = self.rng.choice(np.arange(self.num_clients), size = self.num_workers + border*3, replace = False)
        lossprob = np.sort(candidates[np.argsort(self.local_losses[candidates])[-(self.num_workers + border*2):]])

        candidates = self.rng.choice(lossprob, size = self.num_workers + border, replace = False)
        normprob = candidates[np.argsort(self.local_norms[candidates])[:self.num_workers]]
        self.active_devices = self.comm.bcast(normprob, root = 0)

        if self.rank == 0:
            f = open("l.txt", "a")
            f.write("epoch %3d: " %(epoch_id))
            for i in range (len(lossprob)):
                f.write("%3d " %(lossprob[i]))
            f.write("\n")
            f.close()

            f = open("n.txt", "a")
            f.write("epoch %3d: " %(epoch_id))
            for i in range (len(normprob)):
                f.write("%3d " %(normprob[i]))
            f.write("\n")
            f.close()

            f = open("w.txt", "w")
            for i in range (self.num_clients):
                f.write("client %3d: loss: %10.7f norm: %10.7f\n" %(i, self.local_losses[i], self.local_norms[i]))
            f.close()
        return self.active_devices

    def power_of_choice3 (self, epoch_id):
        border = 0

        candidates = self.rng.choice(np.arange(self.num_clients), size = self.num_candidates, replace = False)
        lossprob = np.sort(candidates[np.argsort(self.local_losses[candidates])[-(self.num_workers + border):]])
        normprob = np.sort(candidates[np.argsort(self.local_norms[candidates])[:(self.num_workers + border)]])
        common = np.sort(np.intersect1d(lossprob, normprob))
        if len(common) > self.num_workers:
            #common = np.sort(common[np.argsort(self.local_losses[common])[-self.num_workers:]])
            common = self.rng.choice(common, size = self.num_workers, replace = False)
        num_extra = self.num_workers - len(common)
        if num_extra > 0:
            candidates = lossprob[~np.isin(lossprob, common)]
            #extra = np.sort(candidates[np.argsort(self.local_losses[candidates])[-num_extra:]])
            extra = self.rng.choice(candidates, size = num_extra, replace = False)
            clients = np.sort(np.concatenate((common, extra), axis=0))
        else:
            clients = common
        self.active_devices = self.comm.bcast(clients, root = 0)

        if self.rank == 0:
            f = open("c.txt", "a")
            f.write("epoch %3d (%3d): " %(epoch_id, len(common)))
            for i in range (len(common)):
                f.write("%3d(%6.4f) " %(common[i], self.local_losses[common[i]]))
            f.write("\n")
            f.close()

            if num_extra > 0:
                f = open("e.txt", "a")
                f.write("epoch %3d (%3d): " %(epoch_id, len(extra)))
                for i in range (len(extra)):
                    f.write("%3d(%6.4f) " %(extra[i], self.local_losses[extra[i]]))
                f.write("\n")
                f.close()

            f = open("l.txt", "a")
            f.write("epoch %3d: " %(epoch_id))
            for i in range (len(lossprob)):
                f.write("%3d " %(lossprob[i]))
            f.write("\n")
            f.close()

            f = open("n.txt", "a")
            f.write("epoch %3d: " %(epoch_id))
            for i in range (len(normprob)):
                f.write("%3d " %(normprob[i]))
            f.write("\n")
            f.close()

            f = open("w.txt", "w")
            for i in range (self.num_clients):
                f.write("client %3d: loss: %10.7f norm: %10.7f\n" %(i, self.local_losses[i], self.local_norms[i]))
            f.close()
        return self.active_devices

    def leadership (self, epoch_id):
        r = 256
        weights = np.ones((self.num_clients))
        b = 4
        lossprob = np.sort(np.argsort(self.local_losses)[-(self.num_workers + 2*b):])
        normprob = np.sort(lossprob[np.argsort(self.local_norms[lossprob])[:(self.num_workers + b)]])
        weights[lossprob] *= r
        weights[normprob] *= r
        weights /= sum(weights)

        self.active_devices = self.rng.choice(np.arange(self.num_clients), size = self.num_workers, replace = False, p = weights)
        if self.rank == 0:
            f = open("l.txt", "a")
            f.write("epoch %3d: " %(epoch_id))
            for i in range (len(lossprob)):
                f.write("%3d " %(lossprob[i]))
            f.write("\n")
            f.close()

            f = open("n.txt", "a")
            f.write("epoch %3d: " %(epoch_id))
            for i in range (len(normprob)):
                f.write("%3d " %(normprob[i]))
            f.write("\n")
            f.close()

            f = open("w.txt", "w")
            for i in range (self.num_clients):
                f.write("client %3d: loss: %10.7f norm: %10.7f weight: %13.10f\n" %(i, self.local_losses[i], self.local_norms[i], weights[i]))
            f.close()
        self.active_devices = self.comm.bcast(self.active_devices, root = 0)

        return self.active_devices

    def update_loss (self, aggregated_losses):
        for i in range (len(aggregated_losses)):
            client_id = self.active_devices[i]
            self.local_losses[client_id] = aggregated_losses[i]

    def update_norm (self, aggregated_norms):
        for i in range (len(aggregated_norms)):
            client_id = self.active_devices[i]
            self.local_norms[client_id] = aggregated_norms[i]
            accum = self.avg_norms[client_id] * self.num_updates[client_id] + self.local_norms[client_id]
            self.num_updates[client_id] += 1
            self.avg_norms[client_id] = accum / self.num_updates[client_id]

    def reset (self):
        self.local_losses = np.full((self.num_clients), np.Inf)
        self.fixed_losses = np.full((self.num_clients), np.Inf)
        self.local_norms = np.zeros((self.num_clients))
        self.avg_norms = np.zeros((self.num_clients))
        self.num_updates = np.zeros((self.num_clients))
