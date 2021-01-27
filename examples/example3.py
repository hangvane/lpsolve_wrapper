# http://web.mit.edu/lpsolve/doc/Python.htm
# max 4x1 + 2x2 + x3
# s.t. 2x1 + x2 <= 1
# x1 + 2x3 <= 2
# x1 + x2 + x3 = 1
# x1 >= 0
# x1 <= 1
# x2 >= 0
# x2 <= 1
# x3 >= 0
# x3 <= 2

import lpsolve_wrapper as lw

model = lw.Model(
    notations={
        'x': lw.notation(
            shape=3,
            upper_bound=[1, 1, 2],
            lower_bound=0,
        )
    })
model.add_constr(
    coefs=[
        lw.coef('x', 2, idx=0),
        lw.coef('x', 1, idx=1),
    ],
    right_value=1,
    constr_type=lw.LEQ
)
model.add_constr(
    coefs=[
        lw.coef('x', 1, idx=0),
        lw.coef('x', 2, idx=2),
    ],
    right_value=2,
    constr_type=lw.LEQ
)
model.add_constr_mat(
    coef_mats={
        'x': [1, 1, 1]
    },
    right_value=1,
    constr_type=lw.EQ
)
objective, notation_list = model.lp_solve(
    obj_func={
        'x': [4, 2, 1],
    },
    minimize=False
)
print('objective:', objective)
print('notations:', notation_list)
