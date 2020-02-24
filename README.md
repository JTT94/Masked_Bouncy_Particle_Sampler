# Masked_BPS_UAI_2020

## Set-Up

The code relies on Apache arrow, hence may only run on Linux. One must also ensure number of actor classes does not exceed number of available processors on server.

The Masked BPS is a distributed computing algorithm and hence performance is dependent on hardware, number of processors available and capacity of network. 

- Install hamiltorch for HMC comparison
`pip install git+https://github.com/AdamCobb/hamiltorch.git`

- Install requirements using pip
`pip install -r requirements.txt`

WARNING: Running experiments serializes output in the order of magnitude of 10s of GB.
