from collections import Iterable

import numpy as np
from lpsolve55 import lpsolve, IMPORTANT

INT_TYPE = 1
FLOAT_TYPE = 0
EQ = 3
GEQ = 2
LEQ = 1
INF = 100000


# TODO: support add_constraintex
def notation(shape=1, upper_bound=None, lower_bound=None, ntype=FLOAT_TYPE):
    """
    Return a dict that represents a new notation.

    :param shape : int or sequence of int, optional, default: 1
        Shape of the new notation, e.g., ``(2, 3)`` or ``2``.
    :param upper_bound : number, vector with the same shape as `shape`, optional, default: None
        The upper bound of the new notation. e.g., `1.1`.
    :param lower_bound : number, vector with the same shape as `shape`, default: None
        The lower bound of the new notation. e.g., `0`.
    :param ntype: {INT_TYPE, FLOAT_TYPE}, optional, default: FLOAT_TYPE
        The integer of float type of the notation.
    """
    return {
        'shape': shape,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'type': ntype
    }


def coef(name, value, idx=0):
    """
    Return a dict that represents a new coefficient.

    :param name : int or sequence of int, optional, default: 1
        Shape of the new notation, e.g., ``(2, 3)`` or ``2``.
    :param value : number, vector with the same shape as `shape`, optional, default: None
        The upper bound of the new notation. e.g., `1.1`.
    :param idx : int, Tuple, optional, default: 0
        It specifies a single array element location, e.g. `0`, `(2,3)`.
    """
    return {
        'name': name,
        'idx': idx,
        'value': value
    }


class Model:
    def __init__(self, notations):
        """
        Init a linear program model and its notations.

        :param notations : dict
            The dict that contains the definitions of notations, e.g., ``{'x': lw.notation()}``.
        notations: {
         `notation name`: {
          'shape': `the shape of notation`,
          'type': `{INT_TYPE, FLOAT_TYPE}`,
          'upper_bound': `the upper bound of the new notation`,
          'lower_bound': `the lower bound of the new notation`,
          '_length': [new] `the number of values that this notation contains`
          '_offset': [new] `the index offset of this notation in the coefficient matrix of the program`
          '_mat': [new] `the coefficient(s) of this notation in a single constraint (not flattened)`
        }
        `lw.notation()` is recommended for generation.
        """

        self.notations = notations

        # the number of values that all notations contain
        length = 0

        # Stores the indexes of integer notation values in the coefficient matrix of the program (1 based indexes).
        must_be_int = []

        for n in notations:
            # Restore the abbreviated `upper_bound` and `upper_bound` to an array.
            for bound_type in ('upper_bound', 'lower_bound'):
                if not isinstance(notations[n][bound_type], Iterable):
                    tmp = np.empty(notations[n]['shape'])
                    tmp.fill(notations[n][bound_type])
                    self.notations[n][bound_type] = tmp

            # Init `_mat`, which helps building constraints, and is reinitialized after building.
            self.notations[n]['_mat'] = np.zeros(notations[n]['shape'])

            self.notations[n]['_offset'] = length
            self.notations[n]['_length'] = np.prod(notations[n]['shape'])

            length += self.notations[n]['_length']

            must_be_int.extend(
                [notations[n]['type'], ] * np.prod(notations[n]['shape'])
            )

        self.lp = lpsolve('make_lp', 0, length)
        lpsolve('set_int', self.lp, must_be_int)
        lpsolve(
            'set_lowbo',
            self.lp,
            np.concatenate([np.ravel(n['lower_bound']) for n in self.notations.values()]).tolist()
        )

        lpsolve(
            'set_upbo',
            self.lp,
            np.concatenate([np.ravel(n['upper_bound']) for n in self.notations.values()]).tolist()
        )
        lpsolve('set_verbose', self.lp, IMPORTANT)

    def _post_proc(self, right_value, constr_type):
        """
        Build a constraint from `self.notations[:]['_mat']` and clean.

        :param right_value : int, float
            The constant on the right side of the constraint.
        :param constr_type : {EQ, LEQ, GEQ}
            The inequality type of the constraint.
        """

        # Flatten the `_mat` of all the notations.
        mats = [n['_mat'].ravel() for n in self.notations.values()]

        # Convert np.inf.
        tmp = np.concatenate(mats)
        tmp[np.isposinf(tmp)] = INF
        tmp[np.isneginf(tmp)] = -INF

        lpsolve('add_constraint', self.lp, tmp.tolist(), constr_type, right_value)

        # Reinitialize `_mat`.
        for n in self.notations.values():
            n['_mat'].fill(0)

    def add_constr(self, coefs, right_value, constr_type):
        """
        Add a constraint by assigning single values.

        :param coefs : list
            The list that contains the definitions of coefficients, e.g., ``[coef('x', 1), coef('y', 2)]``.
        :param right_value : int, float
            The constant on the right side of the constraint.
        :param constr_type : {EQ, LEQ, GEQ}
            The inequality type of the constraint.
        """
        # Assign values to `_mat`.
        for c in coefs:
            self.notations[c['name']]['_mat'].itemset(c['idx'], c['value'])

        self._post_proc(right_value, constr_type)

    def add_constr_mat(self, coef_mats, right_value, constr_type):
        """
        Add a constraint by coefficient mats of notations.

        :param coef_mats : dict
            The dict that contains the coefficients mats of notations, e.g., ``{'x': np.ones(3), 'y': [1, 2, 3]}``.
        :param right_value : int, float
            The constant on the right side of the constraint.
        :param constr_type : {EQ, LEQ, GEQ}
            The inequality type of the constraint.
        """
        # Assign values to `_mat`.
        for name, mat in coef_mats.items():
            self.notations[name]['_mat'] = np.array(mat)

        self._post_proc(right_value, constr_type)

    def add_constr_callback(self, callbacks, right_value, constr_type):
        """
        Add a constraint by callbacks.

        :param callbacks : dict
            The dict that contains the callbacks of notations to assign `_mat`, e.g., ``{'x': lambda x: x[0].fill(1)}``.
        :param right_value : int, float
            The constant on the right side of the constraint.
        :param constr_type : {EQ, LEQ, GEQ}
            The inequality type of the constraint.
        """
        for name, callback in callbacks.items():
            callback(self.notations[name]['_mat'])

        self._post_proc(right_value, constr_type)

    # TODO: fix
    # def gen_mat(self):
    #     """
    #     Return the coefficient mat of **the program**, right side constants, inequality types, integer notation indexes.
    #     """
    #     return self.coef_mat, \
    #            self.right_values, \
    #            self.constr_types, \
    #            self.must_be_int

    def reshape_notation(self, raw_notation):
        """
        Reshape the raw 1-d notation values to the original n-d shaped values for each notations.

        :param raw_notation : list
            The 1-d raw values that got from lp_solve module, representing the optimal value for each notation of the
            program.
        :return dict
            The dict that maps the values of all the notations, e.g., ``{'x': [[1, 2], [3, 4]], 'y': [1, 2, 3]}``.
        """
        nd_results = {}
        for k, v in self.notations.items():
            nd_results[k] = np.reshape(raw_notation[v['_offset']:v['_offset'] + v['_length']], v['shape'])
        return nd_results

    def lp_solve(self, obj_func, minimize, scale=True):
        """
        Solve the program by invoking lp_solve module.

        :param obj_func : dict
            The dict that contains the coefficients mats of notations of the objective function, e.g.,
            ``{'x': np.ones(3), 'y': [1, 2, 3]}``.
        :param minimize : bool
            If True, build a minimize program. Otherwise, build a maximize program.
        :param scale : bool, optional, default: True
            If True, auto scale results, e.g., 0.499999 -> 0.5. Otherwise, no scale results.
        :return `float, dict`
            The float is the objective. The dict maps the optimal values of all the notations, e.g.,
            ``{'x': [[1, 2], [3, 4]], 'y': [1, 2, 3]}``.
        """

        # Assign values to `_mat`.
        for name, mat in obj_func.items():
            self.notations[name]['_mat'] = np.array(mat)

        # Flatten the `_mat` of all the notations.
        mats = [v['_mat'].ravel() for v in self.notations.values()]

        # Convert np.inf.
        tmp = np.concatenate(mats)
        tmp[np.isposinf(tmp)] = INF
        tmp[np.isneginf(tmp)] = -INF

        lpsolve('set_obj_fn', self.lp, tmp.tolist())
        if minimize:
            lpsolve('set_minim', self.lp)
        else:
            lpsolve('set_maxim', self.lp)
        lpsolve('set_scaling', self.lp, 1 if scale else 0)
        lpsolve('solve', self.lp)
        obj = lpsolve('get_objective', self.lp)  # Get the optimal objective.
        raw_notation = lpsolve('get_variables', self.lp)[0]  # Get the 1-d vector of optimal value of notations.
        lpsolve('delete_lp', self.lp)
        # clean
        for v in self.notations.values():
            v['_mat'].fill(0)
        return obj, self.reshape_notation(raw_notation)
