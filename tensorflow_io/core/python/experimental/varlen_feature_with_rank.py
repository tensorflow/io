import tensorflow as tf

class VarLenFeatureWithRank:
  """
  A class used to represent VarLenFeature with rank.
  This allows rank to be passed by users, and when parsing,
  rank will be used to determine the shape of sparse feature.
  User should use this class as opposed to VarLenFeature
  when defining features of data.
  """

  def __init__(self, dtype: tf.dtypes.DType, rank: int=1):
    self.__dtype = dtype
    self.__rank = rank

  @property
  def rank(self):
    return self.__rank

  @property
  def dtype(self):
    return self.__dtype