import numpy as np

from temporal_series_clustering.storage.clusters_history import ClustersHistory


class EpsilonValues:
    """
    Class storing the epsilon values and its related features.

    Structure of each epsilon is:
     - "instant": storing the information of clusters per each instant

     - "intra_mean": storing the mean of the intra_mean of all the clusters for all the instants

     - "inter_mean": storing the inter mean of the combination of all those nodes not belonging to the same cluster for
     all the instants

     - "historical_info_used": storing the instant on which the historical information from previous instants have been
     used
    """

    def __init__(self):
        # In this dictionary all the information per each instant will be stored
        self._info = {}

    def get_all_info_on_epsilon(self, epsilon: str):
        epsilon_info = None
        if epsilon in self._info:
            epsilon_info = self._info[epsilon]

        return epsilon_info

    def insert_all_info_on_epsilon(self, epsilon: str, clusters_history: ClustersHistory, historical_info_used: list):
        self._info[epsilon] = {
            'instant': clusters_history.info,
            'intra_mean': clusters_history.get_all_instants_mean('intra_mean'),
            'inter_mean': clusters_history.get_all_instants_mean('inter_mean'),
            'min_inter_mean': clusters_history.get_all_instants_mean('min_inter_mean'),
            'historical_info_used': historical_info_used
        }

    def get_avg_num_clusters(self) -> dict:
        """
        Get average number of clusters without considering outliers per epsilon

        :return: dict with average number of clusters per epsilon
        :rtype: dict
        """
        return {k: np.mean([len([item for item in list(v['instant'][instant]['value'].keys()) if item != 'outliers'])
                            for instant, value in v["instant"].items()])
                for k, v in self._info.items()}

    def get_num_clusters(self) -> dict:
        """
        Get number of clusters without considering outliers per epsilon

        :return: dict with number of clusters per epsilon
        :rtype: dict
        """
        return {k: [len([item for item in list(v['instant'][instant]['value'].keys()) if item != 'outliers'])
                            for instant, value in v["instant"].items()]
                for k, v in self._info.items()}

    def get_avg_silhouette_score(self) -> dict:
        """
        Get average silhouette score per epsilon

        :return: dict with silhouette score per epsilon
        :rtype: dict
        """
        return {k: np.mean([value['silhouette_score'] for instant, value in v["instant"].items()])
                for k, v in self._info.items()}

    def get_instants_intra_mean(self):
        """
        Get instantaneous intra mean per epsilon

        :return: dict with instantaneous intra mean per epsilon
        :rtype: dict
        """
        return {k: [value['intra_mean'] for instant, value in v["instant"].items()]
                for k, v in self._info.items()}

    def get_instants_inter_mean(self):
        """
        Get instantaneous inter mean per epsilon

        :return: dict with instantaneous inter mean per epsilon
        :rtype: dict
        """
        return {k: [value['inter_mean'] for instant, value in v["instant"].items()]
                for k, v in self._info.items()}

    def get_avg_intra_mean(self):
        """
        Get average intra mean per epsilon

        :return: dict with average intra mean per epsilon
        :rtype: dict
        """
        return {k: v['intra_mean'] for k, v in self._info.items()}

    def get_avg_inter_mean(self):
        """
        Get average inter mean per epsilon

        :return: dict with average inter mean per epsilon
        :rtype: dict
        """
        return {k: v['inter_mean'] for k, v in self._info.items()}

    @property
    def info(self):
        """Getter of info dict"""
        return self._info

    @info.setter
    def info(self, value):
        """Setter of info dict"""
        self._info = value
