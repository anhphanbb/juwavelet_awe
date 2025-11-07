JuWavelet Documentation
------------------------

This document collects the examples and their output in one document.
The functions generating these images are located in the examples folder.

decompose1d.py
=========================

This example analyses a 1-D signal with 4 different methods to showcase the similarities and differences.


.. image:: images/decompose1d/example_decompose1d.png



.. include:: ../examples/decompose1d.py
   :code: python


decompose2d.py
=========================

This example analyses ALIMA temperature data and show the full decomposition in figure (a) and some reconstruction examples in figure (b).


.. image:: images/decompose2d/example_decompose2d_a.png



.. image:: images/decompose2d/example_decompose2d_b.png



.. include:: ../examples/decompose2d.py
   :code: python


identify_clusters.py
=========================

This cluster uses the ALIMA example to show how the (very simple)
cluster identification algorithm works.


.. image:: images/identify_clusters/example_cluster_00.png



.. image:: images/identify_clusters/example_cluster_01.png



.. image:: images/identify_clusters/example_cluster_02.png



.. image:: images/identify_clusters/example_cluster_04.png



.. image:: images/identify_clusters/example_cluster_06.png



.. image:: images/identify_clusters/example_cluster_07.png



.. image:: images/identify_clusters/example_cluster_09.png



.. image:: images/identify_clusters/example_cluster_11.png



.. image:: images/identify_clusters/example_cluster_17.png



.. image:: images/identify_clusters/example_cluster_29.png



.. include:: ../examples/identify_clusters.py
   :code: python


separate2d.py
=========================

This example shows how an artificial 2-D signal composed of
8 different signals can be analysed using the 2-D decomposition.


.. image:: images/separate2d/example_separate2d.png



.. include:: ../examples/separate2d.py
   :code: python


sst.py
=========================

This example recreates the SST analysis figure from Torrence and Compo (1998).



.. image:: images/sst/example_sst.png



.. include:: ../examples/sst.py
   :code: python


stft.py
=========================

These figures show real and Fourier space representation of
1-D Morlet wavelet and Heisenberg basis functions to showcase
similarities and differences between the CWT and STFT approaches.



.. image:: images/stft/example_1dcomp_wav_stft_a.png



.. image:: images/stft/example_1dcomp_wav_stft_b.png



.. image:: images/stft/example_1dcomp_wav_stft_c.png



.. image:: images/stft/example_1dcomp_wav_stft_d.png



.. include:: ../examples/stft.py
   :code: python


tapering.py
=========================

This shows the implemented tapering functions, which should be
used if the boundary of the analysed data is not close to zero.



.. image:: images/tapering/example_tapering1d.png



.. image:: images/tapering/example_tapering2d.png



.. include:: ../examples/tapering.py
   :code: python


wavelet1d.py
=========================

This plots the 1-D Morlet wavelet.



.. image:: images/wavelet1d/example_wavelet1d.png



.. include:: ../examples/wavelet1d.py
   :code: python


wavelet2d.py
=========================

This plots the 2-D Morlet wavelet.



.. image:: images/wavelet2d/example_2dmorlet_a.png



.. image:: images/wavelet2d/example_2dmorlet_b.png



.. include:: ../examples/wavelet2d.py
   :code: python


