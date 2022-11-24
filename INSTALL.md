# QRAO Installation Guide

This document will walk you through setting up a virtual Python environment, installing the quantum random access optimization (QRAO) prototype and its dependencies, and getting started by interacting with the included Jupyter notebooks.

## Clone the repository

We assume that [git](https://git-scm.com/) is installed.  As a first step, clone the repository and enter the clone's directory:

```sh
$ git clone https://github.com/qiskit-community/prototype-qrao.git
$ cd prototype-qrao
```

The remainder of this document will assume you are in the root directory of the repository.

## Set up a virtual Python Environment

Next, we assume that [Python](https://www.python.org/) 3.7 or higher is installed.  It is recommended to use a Python virtual environment that is dedicated to working with `qrao`.  The steps in the remainder of this tutorial will assume that this environment is activated using either method.

### Option 1: venv (included in Python)

You can create and activate a virtual environment with the following commands:

```sh
$ python3 -m venv venv
$ source venv/bin/activate
```

The first command creates a virtual environment in the `venv` folder of the current directory.  We recommend using this name, as it will be ignored by git (i.e., we have added it to `.gitignore`).

Any time you open a new shell and wish to work with QRAO, you will need to activate it using the second line above.  [If you prefer that the virtual environment be activated automatically any time you enter the directory, you may wish to look into using [direnv](https://direnv.net/) or a similar tool.]

### Option 2: conda (recommended only if it is your tool of choice)

The following commands create and activate a conda virtual environment named `qrao_env`:

```sh
$ conda create -n qrao_env python=3
$ conda activate qrao_env
```

Any time you open a new shell and wish to work with QRAO, you will need to activate it using the second line above.

## Install the QRAO prototype

The following command will install the prototype, along with its required dependencies and those of the included notebooks, in "editable" mode (don't forget the `.` after `-e`!). With this, local changes to the source files in your cloned repository will be reflected immediately in the installed environment.

```sh
$ pip install -e '.[notebook-dependencies]'
```

## Testing the Installation (optional)

The QRAO prototype uses [tox](https://github.com/tox-dev/tox) to organize and execute its unit and acceptance tests in a controlled environment.  First, you'll need to install it:

```sh
$ pip install tox
```

You can run every test case on every available version of Python as follows:

```sh
$ tox
```

Details on using tox and the different "environments" defined in `tox.ini` are provided in [`tests/README.md`](tests/README.md).

## Using the Notebooks

First, install Jupyter into the virtual environment:

```sh
$ pip install notebook
```

Then, start the notebook server.  From the root directory:

```sh
$ jupyter notebook
```

Make sure the notebook server is started from the same Python environment (venv or conda) from which you ran `pip install -e '.[notebook-dependencies]'`; otherwise, it may not find `qrao` in the path.

Once the notebook server starts, it will provide a URL for accessing it through your web browser.  To find the tutorial notebooks, navigate in the browser to `docs` and then `tutorials`.
