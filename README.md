# SciPozyx
Using SciPy for filters with Pozyx. 

Matt Fleetwood
10 - 21 - 2017
Portland, OR

Testing of SciPy filters with noisy data from the Pozyx remote sensing system. In particular, the Wiener function is tested on position and time data. The input of the function is postion and time. The output of the function is a filtered velocity. 

Wiener source code used:

https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/signaltools.py#L441
