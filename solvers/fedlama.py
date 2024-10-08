import tensorflow as tf
import numpy as np
import math
from mpi4py import MPI
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean

class FedLAMA:
    def __init__ (self, num_classes, num_workers, average_interval, model, phi):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.num_local_workers = num_workers // self.size
        self.average_interval = average_interval
        self.phi = phi
        self.local_optimizers = []
        for i in range (self.num_local_workers):
            self.local_optimizers.append(SGD(momentum = 0.9))
        num_params = len(model.trainable_variables)
        self.total_size = 0
        for i in range (num_params):
            param = model.trainable_variables[i]
            param_type = param.name.split("/")[1].split(":")[0]
            if param_type == "kernel":
                self.total_size += np.prod(param.shape)
        self.num_comms = np.zeros(num_params)
        self.param_dists_eff = np.zeros(num_params)
        self.param_dists = np.zeros(num_params)
        self.param_intervals = np.zeros(num_params)
        for i in range (len(self.param_intervals)):
            self.param_intervals[i] = 1
        if self.num_classes == 1:
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        else:
            self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
        if self.rank == 0:
            print ("FedLAMA is the optimizer!")
        self.param_types = []
        for i in range (len(model.trainable_variables)):
            param = model.trainable_variables[i]
            self.param_types.append(param.name.split("/")[1].split(":")[0])

    @tf.function
    def cross_entropy_batch(self, label, prediction):
        cross_entropy = self.loss_object(label, prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    def round (self, round_id, models, datasets, local_id):
        model = models[local_id]
        dataset = datasets[local_id]
        optimizer = self.local_optimizers[local_id]

        lossmean = Mean()
        for i in range(self.average_interval):
            images, labels = dataset.next()
            loss, grads = self.local_train_step(model, optimizer, images, labels)
            lossmean(loss)
        return lossmean

    def local_train_step (self, model, optimizer, data, label):
        with tf.GradientTape() as tape:
            prediction = model(data, training = True)
            loss = self.cross_entropy_batch(label, prediction)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, grads

    def average_model (self, checkpoint, epoch_id):
        # Trainable variables.
        for i in range (len(checkpoint.models[0].trainable_variables)):
            if (epoch_id + 1) % self.param_intervals[i] == 0:
                # Get the averaged parameters.
                params = []
                for j in range (self.num_local_workers):
                    params.append(checkpoint.models[j].trainable_variables[i])
                localsum_param = tf.math.add_n(params)
                average_param = self.comm.allreduce(localsum_param, op = MPI.SUM) / self.num_workers

                # Update the param distance.
                dist_sum = 0
                for j in range (self.num_local_workers):
                    param = checkpoint.models[j].trainable_variables[i]
                    dist = (np.linalg.norm(average_param.numpy() - param))**2
                    dist_sum += dist
                dist = self.comm.allreduce(dist_sum, op = MPI.SUM) / self.num_workers

                self.param_dists_eff[i] = dist / (self.param_intervals[i] * self.average_interval * np.prod(average_param.shape))
                self.param_dists[i] = dist / (self.param_intervals[i] * self.average_interval)
                self.num_comms[i] += 1
                # Re-distribute the averaged params.
                for j in range (self.num_local_workers):
                    checkpoint.models[j].trainable_variables[i].assign(average_param)

        # Adjust the update intervals at all the layers again.
        if (epoch_id + 1) % self.phi == 0:
            if epoch_id > 5:
                self.update_intervals(checkpoint)

    def update_intervals (self, checkpoint):
        num_params = len(checkpoint.models[0].trainable_variables)
        for i in range (num_params):
            self.param_intervals[i] = 1

        # Sort the layers based on their contribution to the model discrepancy.
        sorted_index = np.argsort(self.param_dists_eff)
        accum_lambda = 0
        accum_delta = 0
        less_critical_params = []
        for i in range (num_params):
            index = sorted_index[i]
            if self.param_types[index] != "kernel":
                continue
            param = checkpoint.models[0].trainable_variables[index]
            param_size = np.prod(param.shape)

            accum_delta += self.param_dists_eff[index] * param_size
            delta_l = accum_delta / sum(self.param_dists)

            accum_lambda += param_size
            lambda_l = accum_lambda / self.total_size
            less_critical_params.append(index)
            if (1 - lambda_l) < delta_l:
                break
        # Adjust their interval.
        for i in range (num_params):
            if i in less_critical_params:
                self.param_intervals[i] = self.phi
