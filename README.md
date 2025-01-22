# Computational-Optimal-transport
Implementation of the Nys-Sink algorithm presented in https://proceedings.neurips.cc/paper_files/paper/2019/file/f55cadb97eaff2ba1980e001b0bd9842-Paper.pdf.
It is composed by several subroutines:
  * AdaptiveNystrom: adaptive version of Nystrom method that uses doubling trick to get adaptivity and data sampling via approximate ridge score;
  * Sinkhorn algorithm;
  * Round method: to get a feasible solution that lies in the $\ell_1$-ball with center the input coupling.

These subroutines are implemented in utils.py, while Nys-Sink is effectively implemented in in Nys-Sink.ipynb and tested.
