# lpsolve_wrapper

A python wrapper for the module lp_solve, which provides notation based api.

## Features

- Notation management support.
- Offset management support.
- Auto flattening/reshaping.
- Multiple constraint addition methods support.
- Python 2.x/3.x support.

## Install
We strongly recommend you to install `lpsolve` by `conda`:

```commandline
conda install -c conda-forge lpsolve55
```

Then add [lpsolve_wrapper.py](lpsolve_wrapper.py) to your projects.

## Tutorial

There are several [examples](examples). These display some functionality of the interface and can serve as an entry
point for writing more complex code. The following steps are always required when using the interface:

1) It is necessary to import lpsolve_wrapper in your code:

```python
import lpsolve_wrapper as lw
```

2) Create a solver instance and add two notations:

```python
model = lw.Model(
    notations={
        'x': lw.notation(
            lower_bound=0,
        ),
        'y': lw.notation(
            lower_bound=0,
        )
    })
```

3) Access the methods in the `lpsolve_wrapper.py` file, e.g.:

```python
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
```

lpsolve_wrapper provides 3 methods to add a constraint:

1) Add a constraint by assigning single values:

```python
model.add_constr(
    coefs=[
        lw.coef(name='x', idx=[2, 3], value=1),
        lw.coef(name='y', idx=[0], value=2),
    ],
    right_value=75,
    constr_type=lw.GEQ
)
```

2) Add a constraint by coefficient mats of notations:

```python
model.add_constr_mat(
    coef_mats={
        'x': [1, 1, 1],
        'y': [[1, 2], [3, 4]]
    },
    right_value=1,
    constr_type=lw.EQ
)
```

3) Add a constraint by callbacks:

```python
def tmp_func(y):
    y[2][3] = 1


model.add_constr_callback(
    callbacks={
        'x': lambda x: x[0].fill(1),
        'y': tmp_func

    },
    right_value=1,
    constr_type=lw.EQ,
)
```

Use cases can be found in [examples](examples).

  \-    | add_constr() | add_constr_mat() | add_constr_callback()
:---: | :---: | :---: | :---: |
[Example 0](examples/example0.py)  | ✔ |   |   |
[Example 1](examples/example1.py)  | ✔ |   |   |
[Example 2](examples/example2.py)  | ✔ |   |   |
[Example 3](examples/example3.py)  | ✔ | ✔ |   |
[Example 4](examples/example4.py)  |   | ✔ |   |
[Example 5](examples/example5.py)  |   |   | ✔ |