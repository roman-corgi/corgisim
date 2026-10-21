How to simulate an observation (serial or parallel)
===================================================

An observation is a list of visits, and each visit is a sequence of frames. The frames of a visit do not depend on each other, so ``corgisim`` can spread them over several processes. There are three modes:

- ``serial``: everything in one process. Use it for debugging, or when you do not have enough CPUs or memory for a parallel run. It is the slowest mode, but it is the easiest to set up and run.
- ``mp``: worker processes on one machine (a workstation or one cluster node).
- ``mpi``: worker ranks on one or more cluster nodes, launched with ``mpirun``.

With the same seed, all three modes give the same frames, so a parallel run can be checked against a serial one (see :ref:`parallel-seeds`).

``examples/generate_visits_example.py`` and ``examples/generate_visits_from_cpgs_example.py`` run an observation in any of the three modes. This page explains how to choose a mode, how to size the CPUs, memory and workers, how to use the same machinery in your own script, and what can go wrong. 

The numbers on this page come from the default observation of both example scripts (``test/test_data/cpgs_ref_wfov_satspots.xml``), run on a SLURM cluster with 32-core nodes.


Quick start
-----------

Run from ``examples/``. ``--option`` picks the mode and ``-n`` is the number of workers:

.. code-block:: bash

   python generate_visits_example.py --option serial --seed 1
   python generate_visits_example.py --option mp -n 6 --seed 1
   mpirun -np 8 python generate_visits_example.py --option mpi -n 7 --seed 1

.. important::

   Before you pick ``-n``, read :ref:`parallel-sizing`: on a full frame each worker needs about 13 GB of memory.


Which mode to use
-----------------

Measured on the default observation (107 frames, 28 PSFs) on 32-core nodes:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Mode
     - Wall time
     - Speedup
     - Cores
     - Memory reserved
   * - serial
     - 1 h 41
     - 1x
     - 7
     - 22 GB
   * - mp, ``-n 6``
     - 34 min
     - 2.95x
     - 28
     - 100 GB
   * - MPI, 2 nodes, 7 workers
     - 23.5 min
     - 4.3x
     - 56
     - 2 x 60 GB

In ``cpgs_ref_wfov_satspots.xml``, every visit has satellite spots, so each visit has 4 PSF states (satellite spot background, positive, negative, and science). A visit with no satellite spots has only 1 state. With this input, no mode runs more than 4 PSFs at once, and the PSF round takes most of the run time, so extra MPI ranks mostly sit idle and only shorten the detector round (see :ref:`parallel-rounds`).

- **mp is the simplest choice for observations like this one.** It is about 3x faster than serial, on a single node.
- **MPI pays off when more PSF states can run at once than one node holds.** That happens with a time series (one PSF per frame) or with many optics states per visit.


.. _parallel-rounds:

How a run is split into tasks
-----------------------------

Visits run one after another. Each visit runs in two rounds, and the next visit starts only when both are done:

1. **PSF round**: one task per optics state. Without satellite spots, that is 1 PSF   state per visit; with them it is 4 (satellite spot background, positive, negative, and science). Each PSF task calls PROPER, which opens one process per wavelength (``nlam``), so a PSF task uses ``nlam`` CPUs.
2. **Detector round**: one task per frame. Each task adds detector noise to one frame on 1 CPU. A full frame briefly needs about **13 GB** of memory (see :ref:`detector-memory`).

The manager then writes the visit's frames to disk and frees them, so it only holds one visit's frames at a time.

The default observation has ``nlam = 7``, satellite spots, and 7 visits. 3 reference-star visits have 9 frames each, and 4 target visits have 20 frames each. That's 28 PSF states and 107 frames in total.

Polarimetry
~~~~~~~~~~~

With a polarimetry prism, one PSF is several PROPER runs, one per polarization state. These run one after another inside the PSF task (in ``CorgiOptics.get_host_star_psf``), not on separate workers. A polarimetry PSF still uses ``nlam`` CPUs, but takes several times longer.

Time series
~~~~~~~~~~~

With a time series (``zindex`` and ``zval_m_i`` in ``generate_observation_sequence``), every frame has its own optics state. Each task is then a PSF followed by its frame, in a single round.

.. warning::

   Time series parallel runs are not tested yet. You have to supply the Zernike values yourself (one row of ``zval_m_i`` per frame): corgisim does not generate them, and neither ``generate_observation_scenario_from_cpgs`` nor the example scripts take ``zindex``. 


What ``-n`` means
-----------------

``-n`` is the number of **workers**, which is the most tasks that run at the same time. A worker runs one task at a time: ``nlam`` processes during a PSF task, 1 frame during a detector task.

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Mode
     - ``-n`` is
     - How to choose it
   * - ``serial``
     - ignored
     -
   * - ``mp``
     - worker processes on this machine
     - a choice: limited by memory in the detector round (see below)
   * - ``mpi``
     - worker ranks: all MPI processes except rank 0, the manager
     - fixed by the launch: ``-n`` = ``mpirun -np`` - 1. In SLURM, use
       ``-n $((SLURM_NTASKS - 1))``. The script stops if they do not match. 


.. _parallel-sizing:

Sizing a job
------------

The busiest round sets each requirement. Without a time series:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Resource
     - Peak
     - Default observation
   * - CPUs
     - PSF round: ``nlam x min(n, PSF states per visit)``
     - 7 x 4 = 28 ``(7 wvl, 4 PSF states with satellite spots)``
   * - Memory
     - detector round: ~13 GB x min(``n``, frames per visit) + ~2 GB (buffer)
     - ``-n 6``: ~73 GB measured
   * - Useful workers
     - at most the frames of the largest visit; more sit idle
     - 20

.. important::

   With a time series, every task runs PROPER, so CPUs = ``nlam`` x ``n``, and memory is about 13 GB x ``n``. 
   
   On a 32-core node, the CPU limit means at most 4 PSFs at once (4 x 7 = 28 cores). The memory limit on ``-n`` is usually lower than the CPU limit.

.. warning::

   The scripts log the CPUs they see and a suggested ``-n`` before the run
   (:func:`corgisim.parallel_executor.get_cpu_allocation`). The suggestion only considers CPUs and frames, **not memory**. 
   
   For the default observation, it suggests ``-n 20``, which needs about
   260 GB in the detector round. In ``mp`` mode that makes the job hang (see
   :ref:`parallel-hang`). Pick ``-n`` from the memory row instead.


.. _parallel-examples:

The two example scripts
-----------------------

Both scripts run an observation visit by visit, in any of the three modes, and write each visit's frames as full-frame L1 FITS files to a ``V<visit_id>`` folder in ``--output-dir``. They differ in where the observation comes from:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``generate_visits_from_cpgs_example.py``
     - ``generate_visits_example.py``
   * - Observation
     - read from a CPGS file (``--cpgs``)
     - written in Python, in ``setup_observation()``
   * - To change it
     - pass a different file; do not edit the script
     - edit ``setup_observation()`` only
   * - Use it to
     - simulate an observation that is already defined with a CPGS file
     - try settings a CPGS file cannot express, or see how the parts of an observation fit together

The default observation of both scripts is the one in ``test/test_data/cpgs_ref_wfov_satspots.xml``.

.. tip::

   For a quick test, run the CPGS script with
   ``--cpgs ../test/test_data/cpgs_short_sequence.xml`` (2 visits, 18 frames).

Command-line options
~~~~~~~~~~~~~~~~~~~~

Both scripts take the same options, except ``--cpgs``:

- ``--option``: ``serial``, ``mp`` or ``mpi``.
- ``-n``: the number of workers (see :ref:`parallel-sizing`). Ignored in ``serial``.
- ``--seed``: any integer. The same seed gives the same frames; a different seed gives a different noise pattern. Without ``--seed`` the noise changes from run to run (see :ref:`parallel-seeds`).
- ``--output-dir``: where the ``V<visit_id>`` folders and ``run.log`` are written (relative or absolute path). Default: ``examples/output``.
- ``--cpgs`` (CPGS script only): the CPGS XML file to simulate (relative or absolute path).

What you can change in ``generate_visits_example.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the observation setup are in ``setup_observation()``, which returns a dictionary with three parts:

- **Optics** (``cgi_mode``, ``bandpass``, ``optics_keywords``): the coronagraph mode and mask, the DM settings and ``output_dim``. **All visits share these.** Each visit gets its own ``CorgiOptics`` with its own roll angle, but built from the same settings.
- **Stars** (``stars``): one entry per star, each with its scene (``scene.Scene``, where you can add companions and/or 2D scenes), its science detector (``instrument.CorgiDetector``) and its satellite spot settings. Set ``'satspots_are_present': False`` for no satellite spots.
- **Visits** (``visits``): the list of visits, in the order they are taken. Each visit has a ``visit_id``, a ``'star'`` (a key of ``stars``), a ``roll_angle``, ``n_frames`` science frames with ``exp_time`` seconds each, and a ``vistype``.

Rules that are easy to miss:

- **Satellite spots belong to the star, not to the visit.** Every visit of a star with satellite spots gets ``3 x satspots_number_of_frames`` satellite spot frames before its science frames.
- **Keep the** ``visit_id`` **format.** It ends up in the L1 headers and the file names.
- **A different observation needs a different job size.** The number of wavelengths (set by the bandpass and mask), the PSF states per visit (1, or 4 with satellite spots) and the frames in the largest visit set the CPUs, memory and ``-n`` (see :ref:`parallel-sizing`).

.. caution::

   Do not edit the rest of the script: the ``--option`` switch, how the seed is split into per-visit and per-frame seeds, the MPI startup and the frame settings (full frames). To change the noise pattern, pass a different ``--seed`` instead.

The script checks the CPUs against your observation before the run, but it does not check against the memory.

.. note::
   The defaults reproduce ``cpgs_ref_wfov_satspots.xml``, except ``output_dim``: 51 here, 201 when corgisim reads the CPGS file.

What the CPGS script supports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``generate_visits_from_cpgs_example.py`` passes the file to
``observation.generate_observation_scenario_from_cpgs``, which reads it with
``inputs.load_cpgs_data``. From the file, corgisim takes:

- **Stars:** the target, and the reference star if there is one (V magnitude, spectral type, name).
- **Coronagraph and band:** the mask (HLC or wide-FOV SPC for imaging; SPC for spectroscopy, including the rotated version) and the color filter. The DM solution is the one whose contrast is closest to the file's target planet-star flux ratio.
- **Mode per visit:** imaging, polarimetry (``CGIVST_TDD_POL_OBS`` visits, with the visit's ``dpampos`` setting, e.g. ``POL0`` or ``POL45``) or spectroscopy (with the file's slit and its offsets).
- **Detector per star:** EM gain and photon counting, or auto gain.
- **Satellite spots:** if the file asks for them every visit, with their own gain, exposure time and number of frames for the target and the reference star.
- **Visits:** the order, the roll angle, the visit type, and the number of frames and exposure time of each visit.

Some parts are approximations or not supported yet:

- **Auto gain** uses a fixed EM gain of 1000, and an auto-gain visit becomes a single frame of the visit's whole exposure time.
- **Companions** are not read from the file. Add them in ``generate_visits_example.py`` instead.
- **Time series** (``zindex``, speckle variation) are not supported.
- The CPGS path sets ``output_dim`` to 201, and the scripts do not expose it as an option.

Test files with these features are in ``test/test_data``: ``cpgs_ref_pol_satspots.xml`` (polarimetry), ``cpgs_autogain.xml``, ``cpgs_without_reference.xml`` and
``cpgs_short_sequence.xml``.

Before the run, the script reads the file to count the wavelengths, the PSFs per visit and the frames in the largest visit, and logs the CPUs it may use and a suggested ``-n`` for that file. The suggestion ignores memory (see :ref:`parallel-sizing`).


Setting up each mode
--------------------

The example SLURM scripts are in ``examples/``: ``run_visits_*.sh`` for
``generate_visits_example.py`` and ``run_cpgs_*.sh`` for ``generate_visits_from_cpgs_example.py``.
Submit them from ``examples/``. Each script copies the PROPER prescription files
(``roman_preflight.py``, ``roman_preflight_compact.py``) into its working directory, because PROPER
loads the prescription by module name.

.. important::

   Replace ``<EMAIL_ADDRESS>``, ``<your conda environment>`` and, in the MPI scripts, ``<your MPI module>`` with your own. The ``#SBATCH`` lines are for SLURM: adapt them if your cluster uses another scheduler.

Serial: one process
~~~~~~~~~~~~~~~~~~~

One process, one PSF state at a time: ``-c`` = ``nlam``. The memory has to fit one PSF or one frame, whichever is larger.

.. literalinclude:: ../../examples/run_visits_serial.sh
   :language: bash

mp: multiprocessing on one machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``-c`` for the PSF round and ``-n`` for the memory:

- ``-c`` = ``nlam`` x min(``n``, PSFs per visit). That is 28 for the default observation.
- ``-n`` = (``--mem`` - 2 GB) / 13 GB, and no more than the frames of the largest visit. With
  ``--mem=100G`` that's at most 7; the script uses 6.

.. literalinclude:: ../../examples/run_visits_mp.sh
   :language: bash

MPI: one or more cluster nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each rank is its own process with its own CPUs, so the CPU check runs per rank:

- ``--cpus-per-task`` = ``nlam``, so each rank can run one PSF.
- ``--ntasks`` = workers + 1 manager, and ``-n $((SLURM_NTASKS - 1))``.
- ``--ntasks-per-node`` = node cores / ``nlam``. That is 4 on a 32-core node.
- ``--mem`` is **per node**, shared by that node's ranks: ranks per node x 13 GB + margin. That is 60G for 4 ranks.

.. literalinclude:: ../../examples/run_visits_mpi.sh
   :language: bash

.. important::

   The ``mpi4py`` in your environment has to be built against the same MPI library that ``mpirun`` comes from, which is the one you load with ``module load``.


.. _parallel-seeds:

Reproducible frames
-------------------

Detector noise is random, so the frames only match between runs if you pass a seed: ``--seed`` in the example scripts, or ``seed=`` in ``generate_observation_sequence`` and ``generate_observation_scenario_from_cpgs``.

The seed is split in two steps:

1. Each visit gets its own sub-seed (``numpy.random.SeedSequence.spawn``), so visits do not repeat each other's noise.
2. Each frame in the visit gets its own seed from the visit's sub-seed and the frame's position in the sequence (:func:`corgisim.observation.frame_seeds`).

A frame's seed does not depend on which worker runs it, so serial, ``mp`` and ``mpi`` runs with the same seed give the same frames, whatever ``-n`` is.

Without a seed, each frame draws from numpy's global random state in whichever process runs it. The noise statistics are still correct, but the noise pattern changes from run to run, and a parallel run cannot be compared frame by frame with a serial one.


Using parallel runs in your own script
--------------------------------------

The example scripts only choose a **frame runner** and pass it to corgisim.
``generate_observation_sequence`` and ``generate_observation_scenario_from_cpgs`` both take
``frame_runner=``: a function ``frame_runner(func, args_list)`` that returns
``[func(*args) for args in args_list]`` in order. corgisim calls it once for the PSF round and once
for the detector round of each visit. ``None`` (the default) runs everything serially.

For ``mp``, use :func:`corgisim.parallel_executor.run_parallel`:

.. code-block:: python

   import functools
   from corgisim import observation
   from corgisim.parallel_executor import run_parallel

   if __name__ == "__main__":   # needed: the workers are started with "spawn" and re-import this file
       frame_runner = functools.partial(run_parallel, n_jobs=6)
       frames = observation.generate_observation_sequence(
           scene, optics, detector, exp_time, n_frames,
           frame_runner=frame_runner, seed=1)

For ``mpi``, rank 0 is the manager and all other ranks are workers.
:func:`corgisim.mpi_runtime.initialize_mpi_comm` sends the worker ranks into a loop that waits for
tasks, and only returns on rank 0. Always stop the workers at the end, even if the run fails:

.. code-block:: python

   import functools
   from corgisim import observation
   from corgisim.mpi_runtime import initialize_mpi_comm, run_tasks_mpi, shutdown_workers

   comm = initialize_mpi_comm()   # worker ranks never return from this call
   try:
       frame_runner = functools.partial(run_tasks_mpi, comm)
       frames = observation.generate_observation_sequence(
           scene, optics, detector, exp_time, n_frames,
           frame_runner=frame_runner, seed=1)
   finally:
       shutdown_workers(comm)

In both modes the PROPER prescription files have to be in the working directory (the example
scripts call ``roman_preflight_proper.copy_here()``; with MPI, only rank 0 should copy them).
The sizing rules above apply unchanged.

.. seealso::

   :func:`corgisim.observation.generate_observation_sequence`,
   :func:`corgisim.observation.generate_observation_scenario_from_cpgs`,
   :func:`corgisim.parallel_executor.run_parallel` and the :mod:`corgisim.mpi_runtime` module in
   the API reference.


Troubleshooting
---------------

.. _parallel-hang:

The job hangs with no error
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``mp`` mode, if the memory limit kills a worker, ``multiprocessing.Pool`` waits forever for its
task. The job stays running, ``run.log`` stops after a ``[_compute_psf]`` line, and no new frames
appear.

Check:

.. code-block:: bash

   sacct -j <jobid> --format=JobID,State,MaxRSS     # OUT_OF_MEMORY on the batch step
   grep -i oom <job err file>
   srun --jobid=<jobid> --overlap ps -u $USER -o pid,stat,etime,%cpu,rss,cmd --forest

In the ``ps`` output, a pool worker with a shorter elapsed time than the others is a replacement
for a killed one. Cancel the job and lower ``-n`` or raise ``--mem``.

An MPI job never creates its output directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This means rank 0 never got past MPI startup. Nothing is printed, and the ranks may all sit at
100% CPU, because many MPI libraries spin while they wait for messages. To check whether MPI
itself works without corgisim, run this in the same job setup:

.. code-block:: bash

   timeout -k 5 120 mpirun -np $SLURM_NTASKS python -c \
       "from mpi4py import MPI; MPI.COMM_WORLD.Barrier(); print('ok', MPI.COMM_WORLD.Get_rank())"

If that hangs too, the problem is the nodes or the network, not the script. A single node that
can't communicate with the others is enough to cause this. Find it by running the test on smaller
sets of nodes, then leave it out with ``#SBATCH --exclude=<node>`` and tell your cluster's
administrators.

With Intel MPI, ``export I_MPI_DEBUG=5`` logs the network provider and where each rank runs, and
``I_MPI_OFI_PROVIDER=tcp`` runs the same test over TCP instead of the fast network.

.. tip::

   To see where a stuck Python process is, run it with ``python -X faulthandler`` and send it
   ``kill -ABRT <pid>``. It prints its stack to stderr before exiting.

The memory report shows more than the limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``seff`` and ``sacct`` add up each process's memory (RSS), so memory shared between processes is
counted once per process. The ``mp`` run above reported 146 GB in a 100 GB job, while its real
peak was 73 GB. A job that really goes over ``--mem`` gets processes killed. Use your cluster's
job monitor for the real peak.

.. _detector-memory:

Why a detector frame needs ~13 GB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``gain_CIC_Q='roman'`` (used whenever emccd_detect picks its fast gain mode, which includes
an EM gain of 200), emccd_detect's ``partial_CIC`` draws one value per pixel per gain stage in a
single call:

.. code-block:: python

   np.random.binomial(n=1, p=gain_CIC_Q, size=array_size*numel_gain_register)

For a full frame that's 1200 x 2200 pixels x 604 gain stages, about 1.6 billion int64 values or
12.7 GB. macOS compresses this mostly-zero array in memory and reports only about 4 GB, so measure
on the cluster.
