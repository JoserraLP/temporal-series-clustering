from temporal_series_clustering.sheaf.sheaf_model import create_simplified_graph
from temporal_series_clustering.storage.consistencies import ConsistenciesHistory


class SimplifiedGraphsHistory:
    """
    Class storing the simplified graphs for all the instants based
    """

    def __init__(self, num_base_vertices: int):
        # In this dictionary all the information per each instant will be stored
        self._info = {}
        self._num_base_vertices = num_base_vertices

    def create_simplified_graphs(self, consistencies_history: ConsistenciesHistory):
        """
        Create a simplified version (graph) of the sheaf model per each instant of simulation

        :param consistencies_history: input sheaf model consistencies values
        :type consistencies_history: ConsistenciesHistory
        :return: None
        """
        # For each instant create the simplified graph based on sheaf model structure
        self._info = {instant:
                          create_simplified_graph(num_vertices=self._num_base_vertices,
                                                  instant_filtration=
                                                  consistencies_history.get_all_info_on_instant(instant))
                      for instant in range(consistencies_history.get_num_instants())}

    def get_simplified_graph_on_instant(self, instant):
        return self._info[instant] if instant in self._info else None

    @property
    def info(self):
        """Getter of info dict"""
        return self._info

    @info.setter
    def info(self, value):
        """Setter of info dict"""
        self._info = value
