import numpy as np
import numpy.typing as npt

def normalization(variables):
    min_val = np.min(variables, axis=0)
    max_val = np.max(variables, axis=0)
    _range = max_val - min_val
    _range[_range <= 0] = 1
    return (variables - min_val) / _range


def create_derived_variables(urm: npt.ArrayLike):
  n_users, n_items = urm.shape

  C_sum_rating = urm.sum(axis=1).getA().flatten()
  I_sum_rating = urm.sum(axis=0).getA().flatten()
  C_quantity = np.ediff1d(urm.tocsr().indptr) # count of each row
  I_quantity = np.ediff1d(urm.tocsc().indptr) # count of each colum
  C_quantity_no_zero = C_quantity.copy()
  C_quantity_no_zero[C_quantity_no_zero == 0] = 1
  I_quantity_no_zero = I_quantity.copy()
  I_quantity_no_zero[I_quantity_no_zero == 0] = 1
  C_aver_rating = C_sum_rating / C_quantity_no_zero
  I_aver_rating = I_sum_rating / I_quantity_no_zero

  I_likability = np.zeros(n_items)
  C_seen_popularity = np.zeros(n_users)
  C_seen_rating = np.zeros(n_users)
  rows, cols = urm.nonzero()
  for row, col in zip(rows, cols):
    I_likability[col] += urm[row, col] - C_aver_rating[row]
    C_seen_popularity[row] += I_quantity[col]
    C_seen_rating[row] += I_aver_rating[col]
  I_likability /= I_quantity_no_zero
  C_seen_popularity /= C_quantity_no_zero
  C_aver_rating /= C_quantity_no_zero

  return C_aver_rating, C_quantity, C_seen_popularity, C_seen_rating,\
         I_aver_rating, I_quantity, I_likability


def create_related_variables(urm, icm, ucm):
    C_aver_rating, C_quantity, C_seen_popularity, C_seen_rating,\
    I_aver_rating, I_quantity, I_likability = create_derived_variables(urm)
    item_related_variables = np.hstack([
        I_aver_rating.reshape((-1, 1)),
        I_quantity.reshape((-1, 1)),
        # I_likability.reshape((-1, 1)),
        icm.toarray(),
    ])
    user_related_variables = np.hstack([
        C_aver_rating.reshape((-1, 1)),
        C_quantity.reshape((-1, 1)),
        # C_seen_popularity.reshape((-1, 1)),
        # C_seen_rating.reshape((-1, 1)),
        ucm.toarray(),
    ])
    item_related_variables = normalization(item_related_variables)
    user_related_variables = normalization(user_related_variables)
    return item_related_variables, user_related_variables
