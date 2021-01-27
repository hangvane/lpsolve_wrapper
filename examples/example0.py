# http://web.mit.edu/lpsolve/doc/Python.htm
# max 143x + 60y
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
print('x:', notation_list['x'])
print('y:', notation_list['y'])
