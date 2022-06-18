- 2022-06-17: Fixed a bug in the "weighted" sampling method for magic
  rounding.  Anyone who did experiments using weighted sampling
  previously might want to try it again, as results ought to be better
  statistically now.
- 2022-06-13: Packages that are _only_ necessary for running the
  included Jupyter notebooks are no longer listed as requirements of
  `qrao`, so that `qrao` can be more easily used as a library.  If you
  wish to install these optional dependencies, run
  `pip install -e '.[notebook-dependencies]'`. ([#34])
- 2022-06-09: Added a comparison to exact optimal function value using
  `CplexOptimizer` in the first tutorial ([#27])

[#27]: https://github.com/qiskit-community/prototype-qrao/pull/27
[#34]: https://github.com/qiskit-community/prototype-qrao/pull/34
[#40]: https://github.com/qiskit-community/prototype-qrao/pull/40
