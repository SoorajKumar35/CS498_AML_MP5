import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

class acc_classifier:

    def __init__(self, data_dir):
        self.get_activity_data(data_dir)
        self.get_training_and_testing_data()
        self.cluster_data()
        self.train_data_hists = self.quantize_data(self.train_data)
        pass

    def get_activity_data(self, data_dir, segment_size=32):

        # First, we iterate through all the needed directories
        directory_list_w_MODEL, directory_list = os.listdir(data_dir), []
        directory_list = [x for x in directory_list_w_MODEL if 'MODEL' not in x and '.txt' not in x and '.m' not in x and '.DS_Store' not in x]
        sample_data, data = {}, {}
        file_counter = 0
        for sample_dir in directory_list:
            sample_data[sample_dir] = []
            for file in os.listdir(data_dir + sample_dir + '/'):
                file_counter += 1
                # if sample_data[sample_dir] == []:
                #     sample_data[sample_dir] += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
                # else:
                #     # sample_data[sample_dir] = np.concatenate((sample_data[sample_dir], np.genfromtxt(data_dir + sample_dir + '/' + file)), axis=0)
                #     sample_data[sample_dir] += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
                sample_data[sample_dir] += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
            data[sample_dir] = []
            for i in np.arange(len(sample_data[sample_dir])):
                obs_data, new_obs_data = sample_data[sample_dir][i], []
                for j in np.arange(int(obs_data.shape[0]/segment_size)):
                    segment = obs_data[j*segment_size:(j+1)*segment_size, :].flatten()[:, np.newaxis]
                    if new_obs_data == []:
                        new_obs_data = segment
                    else:
                        new_obs_data = np.concatenate((new_obs_data, segment), axis=1)
                data[sample_dir] += [new_obs_data.T]
        # print(file_counter)
        self.data = data
        self.sample_data = sample_data

    def get_training_and_testing_data(self):
        self.train_data, self.test_data, self.clustering_data = {}, {}, []
        for key in self.data.keys():
            train_split = int(0.90 * len(self.data[key]))
            arr = self.data[key][:train_split]
            # if self.clustering_data == []:
            #     self.clustering_data = arr
            # else:
            #     self.clustering_data = np.concatenate((self.clustering_data, arr), axis=0)
            self.clustering_data += arr
            self.train_data[key] = arr
            self.test_data[key] = self.data[key][train_split:]
        clustered_train_data = []
        for i in range(len(self.clustering_data)):
            signal = self.clustering_data[i]
            if clustered_train_data == []:
                clustered_train_data = signal
            else:
                clustered_train_data = np.concatenate((clustered_train_data, signal), axis=0)
        self.clustering_data = clustered_train_data

    def cluster_data(self, num_clusters_1st_lvl=16, num_clusters_2nd_lvl=3):

        # First, we create the initial top level clusters
        KMeans_model_1st_lvl = KMeans(n_clusters=num_clusters_1st_lvl,
                                init='k-means++',
                                random_state=0
                                )
        # We then fit the data to the the first level Kmeans model
        KMeans_model_1st_lvl.fit(self.clustering_data)
        predictions = KMeans_model_1st_lvl.predict(self.clustering_data)
        first_lvl_cluster_centers = {}
        for cluster_num in np.arange(KMeans_model_1st_lvl.n_clusters):
            curr_cluster_data = self.clustering_data[cluster_num == predictions, :]
            KMeans_model_2nd_lvl =  KMeans(n_clusters=num_clusters_2nd_lvl,
                   init='k-means++',
                   random_state=0
                   )
            KMeans_model_2nd_lvl.fit(curr_cluster_data)
            second_lvl_cluster_centers = {}
            for second_lvl_cluster_num in np.arange(KMeans_model_2nd_lvl.n_clusters):
                cluster_center = KMeans_model_2nd_lvl.cluster_centers_[second_lvl_cluster_num]
                second_lvl_cluster_centers[second_lvl_cluster_num] = (second_lvl_cluster_num, cluster_center)
            first_lvl_curr_center = KMeans_model_1st_lvl.cluster_centers_[cluster_num]
            first_lvl_cluster_centers[cluster_num] = (cluster_num, first_lvl_curr_center, second_lvl_cluster_centers)
        self.cluster_dictionary = first_lvl_cluster_centers

    def quantize_data(self, data, num_clusters_1st_lvl=16, num_clusters_2nd_lvl=3):

        quantized_data = {}
        histograms = []
        for act in data.keys():
            for sample in np.arange(len(data[act])):
                quantized_data_sample = np.zeros(num_clusters_1st_lvl * num_clusters_2nd_lvl)
                for i in np.arange(data[act][sample].shape[0]):
                    hist_idx = self.return_closest_cluster_index(data[act][sample][i])
                    quantized_data_sample[hist_idx] += 1
                histograms += [quantized_data_sample]
            quantized_data[act] = histograms
        return quantized_data[act]

    def return_closest_cluster_index(self, data_sample):

        min_idx, min_dist = float("inf"), float("inf")
        for cluster_num, center in self.cluster_dictionary.items():
            dist = np.linalg.norm(data_sample - center[1])
            if dist < min_dist:
                min_idx, min_dist = cluster_num, dist
        min_idx_2, min_dist_2 = float("inf"), float("inf")
        for cluster_num, center in self.cluster_dictionary[min_idx][2].items():
            dist = np.linalg.norm(data_sample - center[1])
            if dist < min_dist_2:
                min_idx_2, min_dist_2 = cluster_num, min_dist_2
        return min_idx * min_idx_2





data_dir = 'HMP_Dataset/'
acc_classifier(data_dir)

# def get_train_and_test_data(self, data_dir):
#
#     # First, we get the needed files in the all subdirectories by pruning out those files that are not required.
#     directory_list_w_MODEL, directory_list = os.listdir(data_dir), []
#     directory_list = [x for x in directory_list_w_MODEL if 'MODEL' not in x and '.txt' not in x and '.m' not in x and '.DS_Store' not in x]
#
#     # We get all samples and then split them into train and test data. We also segment each of them into 32 * 3 size vectors
#     observations, seg_observations = [], []
#     for sample_dir in directory_list:
#         for file in os.listdir(data_dir + sample_dir + '/'):
#             observations += [np.genfromtxt(data_dir + sample_dir + '/' + file)]
#     for obs in range(len(observations)):
#         obs_segmented = []
#         for i in np.arange(int(observations[obs].shape[0]/32)):
#             if obs_segmented == []:
#                 obs_segmented = observations[obs][i*32:(i + 1)*32, :].flatten()[:, np.newaxis]
#             else:
#                 obs_segmented = np.concatenate((obs_segmented, observations[obs][i*32:(i + 1)*32, :].flatten()[:, np.newaxis]), axis=1)
#         seg_observations += [obs_segmented.T]
#     pass
