- 2022-06-13: Packages that are _only_ necessary for running the
  included Jupyter notebooks are no longer listed as requirements of
  `qrao`, so that `qrao` can be more easily used as a library.  If you
  wish to install these optional dependencies, run
  `pip install -e '.[notebook-dependencies]'`. ([#34])

[#34]: https://github.com/qiskit-community/prototype-qrao/pull/34
