import numpy as np

class IDEncoder:

    """
    Encodes excisting ids and handles new ids by mapping them to 'new_value'
    """

    def fit(self, ids):
        """
        Fit id encoder

        :param ids: List of ids
        """

        ids = np.array(ids)

        assert ids.ndim == 1
        self.classes = np.unique(np.append(ids, 'new_value'))
        self.new_value = self.transform(['new_value'])[0]

    def transform(self, ids):
        """
        Transform ids

        :param ids: List of ids
        :return: List of encoded ids
        """
        ids = np.array(ids)
        assert hasattr(self, 'classes')
        assert ids.ndim == 1

        # If empty return empty
        if ids.size == 0:
            return np.array([])

        ids = np.where(np.isin(ids, self.classes, invert=True), "new_value", ids)

        return np.searchsorted(self.classes, ids)

    def inverse_transform(self, ids):

        """
        Transform ids to original encoding.

        :param ids: List of encoded ids
        :return: List of decoded ids
        """

        ids = np.array(ids)
        assert hasattr(self, 'classes')
        assert ids.ndim == 1

        if ids.size == 0:
            return np.array([])

        ids = np.where(np.isin(ids, np.arange(len(self.classes)), invert=True), self.new_value, ids)

        ids = np.asarray(ids)
        return self.classes[ids]
