#### 0.2 (not yet released)

- 2022-11-23: The minimum required version of Qiskit is now 0.37.0,
  and the minimum required version of Qiskit Optimization is now
  0.4.0.  Python 3.6 is no longer supported. ([#49])
- 2022-11-23: The backend tests now use qiskit-ibm-provider rather
  than qiskit-ibmq-provider. ([#15])

#### 0.1 (2022-11-23)

- 2022-06-21: Magic rounding now supports all available encodings.
  ([#33])
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

[#15]: https://github.com/qiskit-community/prototype-qrao/pull/15
[#27]: https://github.com/qiskit-community/prototype-qrao/pull/27
[#33]: https://github.com/qiskit-community/prototype-qrao/pull/33
[#34]: https://github.com/qiskit-community/prototype-qrao/pull/34
[#40]: https://github.com/qiskit-community/prototype-qrao/pull/40
[#49]: https://github.com/qiskit-community/prototype-qrao/pull/49
