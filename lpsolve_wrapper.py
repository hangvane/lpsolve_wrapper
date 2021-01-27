from collections import Iterable

from lp_maker import lp_maker, lpsolve
import numpy as np

INT_TYPE = 1
FLOAT_TYPE = 0
EQ = 0
GEQ = 1
LEQ = -1
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

        # The coefficient matrix of **the program**, where each element represents the left side of a single constraint.
        self.coef_mat = []

        # It stores the right side constants of constraints.
        self.right_values = []

        # Each element = {EQ, GEQ, LEQ} represents the inequality type of a single constraint.
        self.constr_types = []

        # It stores the indexes of integer notation values in the coefficient matrix of the program (1 based indexes).
        self.must_be_int = []

        # the number of values that all notations contain
        self.length = 0

        self.notations = notations

        for n in notations:
            # Restore the abbreviated `upper_bound` and `upper_bound` to an array.
            for bound_type in ('upper_bound', 'lower_bound'):
                if not isinstance(notations[n][bound_type], Iterable):
                    tmp = np.empty(notations[n]['shape'])
                    tmp.fill(notations[n][bound_type])
                    self.notations[n][bound_type] = tmp

            # Init `_mat`, which helps building constraints, and is reinitialized after building.
            self.notations[n]['_mat'] = np.zeros(notations[n]['shape'])

            self.notations[n]['_offset'] = self.length
            self.notations[n]['_length'] = np.prod(notations[n]['shape'])

            self.length += self.notations[n]['_length']

            if notations[n]['type'] == INT_TYPE:
                self.must_be_int.extend(
                    np.arange(self.notations[n]['_length'])
                    + self.notations[n]['_offset'] + 1
                )

    def _post_proc(self, right_value, constr_type):
        """
        Build a constraint from `self.notations[:]['_mat']` and clean.

        :param right_value : int, float
            The constant on the right side of the constraint.
        :param constr_type : {EQ, LEQ, GEQ}
            The inequality type of the constraint.
        """
        self.right_values.append(right_value)
        self.constr_types.append(constr_type)

        # TODO: speed up
        # Flatten the `_mat` of all the notations.
        mats = [n['_mat'].ravel() for n in self.notations.values()]

        # Convert np.inf.
        tmp = np.concatenate(mats)
        tmp[np.isposinf(tmp)] = INF
        tmp[np.isneginf(tmp)] = -INF

        self.coef_mat.append(tmp)

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

    def gen_mat(self):
        """
        Return the coefficient mat of **the program**, right side constants, inequality types, integer notation indexes.
        """
        return self.coef_mat, \
               self.right_values, \
               self.constr_types, \
               self.must_be_int

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

        print('lpsolve wrapper: creating...')
        lp = lp_maker(
            tmp.tolist(),  # n vector of coefficients for a linear objective function
            np.array(self.coef_mat, dtype=np.float32).tolist()  # tolist() when it contains only 1 constraint for debug
            if len(self.coef_mat) == 1 else
            np.array(self.coef_mat, dtype=np.float32),  # coefficient matrix of constraints
            self.right_values,  # right side constants
            self.constr_types,  # inequality types of constraints
            np.concatenate([np.ravel(n['lower_bound']) for n in self.notations.values()]).tolist(),  # lower bounds
            np.concatenate([np.ravel(n['upper_bound']) for n in self.notations.values()]).tolist(),  # upper bounds
            self.must_be_int,  # Vector of integer variables. May be omitted or empty.
            1 if scale else 0,  # Auto scale flag, e.g., 0.499999 -> 0.5. Off when 0 or omitted.
            1 if minimize else 0  # Set maximum lp when this flag equals 0 or omitted.
        )
        print('lpsolve wrapper: solving...')
        lpsolve('solve', lp)
        obj = lpsolve('get_objective', lp)  # Get the optimal objective.
        raw_notation = lpsolve('get_variables', lp)[0]  # Get the 1-d vector of optimal value of notations.

        # clean
        for v in self.notations.values():
            v['_mat'].fill(0)
        return obj, self.reshape_notation(raw_notation)
