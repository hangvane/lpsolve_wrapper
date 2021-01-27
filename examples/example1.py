# http://web.mit.edu/lpsolve/doc/Python.htm
# max 143x + 60y
# 120x + 210y <= 15000
# 110x + 30y <= 4000
# x + y <= 75
# x >= 0, y >= 0

import lpsolve_wrapper as lw

model = lw.Model(
    notations={
        'x': lw.notation(
            lower_bound=0,
        ),
        'y': lw.notation(
            lower_bound=0,
        )
    })

model.add_constr(
    coefs=[
        lw.coef('x', 120),
        lw.coef('y', 210),
    ],
    right_value=15000,
    constr_type=lw.LEQ
)
model.add_constr(
    coefs=[
        lw.coef('x', 110),
        lw.coef('y', 30),
    ],
    right_value=4000,
    constr_type=lw.LEQ
)
model.add_constr(
    coefs=[
        lw.coef('x', 1),
        lw.coef('y', 1),
    ],
    right_value=75,
    constr_type=lw.LEQ
)
objective, notation_list = model.lp_solve(
    obj_func={
        'x': 143,
        'y': 60,
    },
    minimize=False
)
print('objective:', objective)
print('notations:', notation_list)
