# http://web.mit.edu/lpsolve/doc/Python.htm
# P = (110)(1.30)x + (30)(2.00)y + (125)(1.56) = 143x + 60y + 195z
# 120x + 210y + 150.75z <= 15000
# 110x + 30y + 125z <= 4000
# x + y + z <= 75
# x >= 0, y >= 0, z >= 0

import lpsolve_wrapper as lw

model = lw.Model(
    notations={
        'x': lw.notation(
            lower_bound=0,
        ),
        'y': lw.notation(
            lower_bound=0,
        ),
        'z': lw.notation(
            lower_bound=0,
        )
    })
model.add_constr(
    coefs=[
        lw.coef('x', 120),
        lw.coef('y', 210),
        lw.coef('z', 150.75),
    ],
    right_value=15000,
    constr_type=lw.LEQ
)
model.add_constr(
    coefs=[
        lw.coef('x', 110),
        lw.coef('y', 30),
        lw.coef('z', 125),
    ],
    right_value=4000,
    constr_type=lw.LEQ
)
model.add_constr(
    coefs=[
        lw.coef('x', 1),
        lw.coef('y', 1),
        lw.coef('z', 1),
    ],
    right_value=75,
    constr_type=lw.LEQ
)
objective, notation_list = model.lp_solve(
    obj_func={
        'x': 143,
        'y': 60,
        'z': 195,
    },
    minimize=False
)
print('objective:', objective)
print('notations:', notation_list)
