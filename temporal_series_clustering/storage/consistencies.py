class ConsistenciesHistory:
    """
    Class storing the historical clusters of the algorithms
    """

    def __init__(self, info: dict):
        # Pop the output node '@' from all the instants
        for instant, value in info.items():
            # None for checking if the '@' is not in the dict
            value.pop('@', None)

        # In this dictionary all the information per each instant will be stored
        self._info = info

    def get_all_info_on_instant(self, instant: int) -> dict:
        """
        Get all the consistencies information for a given instant. Additionally, output node consistency can be removed

        :param instant: instant where the information will be retrieved
        :type instant: int
        :return: dictionary with the information for a given instant
        :rtype: dict
        """
        instant_info = None
        if instant in self._info:
            instant_info = self._info[instant]
        return instant_info

    def get_num_instants(self) -> int:
        """
        Get the number of all instants

        :return: number of instants stored
        """
        return len(self._info.keys())

    @property
    def info(self):
        """Getter of info dict"""
        return self._info

    @info.setter
    def info(self, value):
        """Setter of info dict"""
        self._info = value
