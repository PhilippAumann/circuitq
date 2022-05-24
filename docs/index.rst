
.. figure:: ../new_logo.png
   :width: 5cm


Welcome to CircuitQ's documentation!
====================================
   CircuitQ is an an open-source toolbox for the analysis of superconducting circuits implemented in Python.
   It features the automated construction of a symbolic Hamiltonian of the input circuit, as well as a dynamic numerical representation of this Hamiltonian with a variable basis choice.
   Additional features include the estimation of the T1 lifetimes of the circuit states under various noise mechanisms.
   The circuit quantization is both applicable to circuit inputs from a large design space and open-source.

To get started, have a look at the `installation instructions <installation.html>`_ and at our `quick demo <transmon_demo.html>`_ for a demonstration of the basic usage of the toolbox.

A more comprehensive description of CircuitQ's functionalities can be found in the `tutorial <tutorial.html>`_.

For further illustrations of the toolbox, please see the `more examples <more_examples.html>`_ section.
The `API reference <api_reference.html>`_ provides an overview of the range of functions and is constructed based on the docstrings.


Table of contents
-----------------
.. toctree::
   :maxdepth: 1

   installation.rst
   transmon_demo.ipynb
   tutorial.ipynb
   more_examples.rst
   api_reference.rst

.. note::
   Please refer to our `preprint on arXiv <http://arxiv.org/abs/2106.05342>`_ for more details on physics and implementation and to cite our project.

The source code can be found on `CircuitQ's repository at GitHub <https://github.com/PhilippAumann/circuitq>`_.

CircuitQ was developed by Philipp Aumann and Tim Menke under the supervision of `William Oliver <https://equs.mit.edu/>`_ and `Wolfgang Lechner <https://www.uibk.ac.at/th-physik/quantum-optimization/>`_.

`Index <genindex.html>`_
------------------------

..
   * :ref:`modindex`
   * :ref:`search`
