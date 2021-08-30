Installation
====================================

Installation via pip
----------------------
CircuitQ can be installed by using python's package manager ``pip``:

.. code-block::

   pip install circuitq


Installation via conda
----------------------
If you are using the Anaconda distribution, you may want to avoid mixing ``pip`` and ``conda``, due to a different handling of dependencies of those installers [1]_. CircuitQ can also be installed using ``conda`` via the conda-forge channel.
You could either first add conda-forge to your installation channels and subsequently install CircuitQ by

.. code-block::

   conda config --add channels conda-forge
   conda install circuitq

or you can state the channel directly for the installation:

.. code-block::
   
   conda install -c conda-forge circuitq

.. [1] The situation improved with conda version 4.6.0 which provides the experimental ``pip_interop_enabled`` feature. However best practice is still to avoid mixing conda and pip if possible. 


Installation via GitHub
------------------------
Alternatively, you can clone `CircuitQ's repository at GitHub <https://github.com/PhilippAumann/circuitq>`_ and
run one of the following commands inside the projects main directory:

.. code-block:: bash

   python setup.py develop

This is recomended for developers as it keeps the package up to date, or:

.. code-block:: bash

   pip install .

