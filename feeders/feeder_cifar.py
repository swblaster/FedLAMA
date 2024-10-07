import os
import math
import pickle
import time
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.python.data.experimental import AUTOTUNE
import tensorflow_datasets as tfds

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=float)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

class cifar:
    def __init__ (self, batch_size, num_workers, num_clients, num_classes, alpha):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.num_workers = num_workers
        self.num_clients = num_clients
        self.num_local_workers = int(num_workers / self.size)
        self.num_local_clients = num_clients // self.size
        self.train_batch_size = batch_size
        self.valid_batch_size = 100
        self.num_train_samples = 50000
        self.num_valid_samples = 10000
        self.num_classes = num_classes
        self.alpha = alpha
        self.devices = []
        self.rng = np.random.default_rng()
        train_data = None
        train_label = None

        # Read training files.
        name = "cifar" + str(self.num_classes)
        dataset = tfds.as_numpy(tfds.load(name = name, split = tfds.Split.TRAIN, batch_size = -1))
        train_data, train_label = dataset["image"], dataset["label"]
        dataset = tfds.as_numpy(tfds.load(name = name, split = tfds.Split.TEST, batch_size = -1))
        valid_data, valid_label = dataset["image"], dataset["label"]

        self.train_data = train_data.astype('float32') / 255.
        self.valid_data = valid_data.astype('float32') / 255.

        self.partitions = {}
        if self.rank == 0:
            for i in range (self.num_clients):
                name = "partitions/cifar" + str(num_classes) + "_" + str(i) + ".txt"
                f = open(name, "r")
                lines = f.readlines()
                partition = []
                for j in range (len(lines)):
                    line = lines[j].split('\n')
                    value = int(line[0])
                    partition.append(value)
                self.partitions[i] = partition
                f.close()
        self.partitions = self.comm.bcast(self.partitions, root = 0)
        if self.rank == 0:
            for i in range (len(self.partitions)):
                print ("worker " + str(i) + " has " + str(len(self.partitions[i])) + " samples")

        # Find the number of local training samples for the local clients.
        self.num_local_samples = []
        for i in range (self.num_clients):
            self.num_local_samples.append(len(self.partitions[i]))

        self.num_local_train_samples = 50000 // self.size
        self.train_sample_offset = self.num_local_train_samples * self.rank
        self.num_valid_batches = self.num_valid_samples // self.valid_batch_size
        self.num_valid_samples = self.num_valid_batches * self.valid_batch_size

        self.per_pixel_mean = np.array(self.train_data).astype(np.float32).mean(axis=0)
        self.per_pixel_std = np.array(self.train_data).astype(np.float32).std(axis=0)

        self.train_label = dense_to_one_hot(train_label, self.num_classes)
        self.valid_label = dense_to_one_hot(valid_label, self.num_classes)
        
    def read_train_image (self, indices):
        info = indices.numpy()
        sample_id = info[0]
        device_id = info[1]
        index = self.partitions[device_id][sample_id]
        image = self.train_data[index]
        image = np.subtract(image, self.per_pixel_mean)
        image = np.divide(image, self.per_pixel_std)
        label = self.train_label[index]
        return image, label

    def read_valid_image (self, sample_id):
        index = sample_id.numpy()
        image = self.valid_data[index]
        image = np.subtract(image, self.per_pixel_mean)
        image = np.divide(image, self.per_pixel_std)
        label = self.valid_label[index]
        return image, label

    def read_hessian_image (self, sample_id):
        index = self.train_sample_offset + sample_id.numpy()
        image = self.train_data[index]
        image = np.subtract(image, self.per_pixel_mean)
        image = np.divide(image, self.per_pixel_std)
        label = self.train_label[index]
        return image, label

    def augmentation(self, x, y):
        x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
        x = tf.image.random_crop(x, (32, 32, 3))
        x = tf.image.random_flip_left_right(x)
        return x, y

    def train_dataset (self, client_id):
        # Client_id should be the global client ID.
        num_samples = self.num_local_samples[client_id]
        client_index = np.full((num_samples, 1), client_id)
        sample_index = np.reshape(np.arange(num_samples), (num_samples, 1))
        indices = np.concatenate((sample_index, client_index), axis = 1)
        dataset = tf.data.Dataset.from_tensor_slices(indices)
        dataset = dataset.shuffle(num_samples, seed = int(time.time()))
        dataset = dataset.map(lambda x: tf.py_function(self.read_train_image, inp = [x], Tout = [tf.float32, tf.float32]))
        dataset = dataset.map(self.augmentation)
        dataset = dataset.batch(self.train_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_samples))
        dataset = dataset.map(lambda x: tf.py_function(self.read_valid_image, inp = [x], Tout = [tf.float32, tf.float32]), num_parallel_calls = AUTOTUNE)
        dataset = dataset.batch(self.valid_batch_size)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def check_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.train_batch_size * 128))
        dataset = dataset.map(lambda x: tf.py_function(self.read_hessian_image, inp = [x], Tout = [tf.float32, tf.float32]), num_parallel_calls = AUTOTUNE)
        dataset = dataset.batch(self.train_batch_size * 128)
        dataset = dataset.repeat()
        return dataset.__iter__()
