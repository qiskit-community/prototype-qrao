# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Setup file for QRAO."""

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="qrao",
    description="Quantum Random Access Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
)
