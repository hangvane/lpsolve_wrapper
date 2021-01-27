# \max \sum_{i=0}^1 \sum_{j=0}^1 x_{i,j} - \sum_{m=0}^2 \sum_{n=0}^2 y_{m,n}
# \sum_{i=0}^1 \sum_{j=0}^1 x_{i,j} <= 1
# \sum_{m=0}^2 \sum_{n=0}^2 y_{m,n} >= 1
# x_{i,j} \in \{0,1\}
# 0 <= y_{m,n} <= 1

import numpy as np
import lpsolve_wrapper as lw

model = lw.Model(
    notations={
        'x': lw.notation(
            shape=(2, 2),
            upper_bound=1,
            lower_bound=0,
            ntype=lw.INT_TYPE,
        ),
        'y': lw.notation(
            shape=(3, 3),
            upper_bound=1,
            lower_bound=0,
            ntype=lw.FLOAT_TYPE,
        )
    })
model.add_constr_mat(
    coef_mats={
        'x': np.ones(shape=(2, 2))
    },
    right_value=1,
    constr_type=lw.LEQ
)
model.add_constr_mat(
    coef_mats={
        'y': np.ones(shape=(3, 3))
    },
    right_value=1,
    constr_type=lw.GEQ
)
objective, notation_list = model.lp_solve(
    obj_func={
        'x': np.ones(shape=(2, 2)),
        'y': np.ones(shape=(3, 3)) * -1,
    },
    minimize=False
)
print('objective:', objective)
print('notations:', notation_list)
