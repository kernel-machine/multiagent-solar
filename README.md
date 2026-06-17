# Energy-Aware Decentralized Multi Agent Deep Reinforcement Learning Scheduler for Energy Harvest Devices

# Authors:
- Author
- Author
- Author

# Abstract
Collaborative edge devices powered by solar energy face the challenge of operating under intermittent and unpredictable energy availability. This paper proposes a decentralized Multi-Agent Deep Reinforcement Learning (MADRL)-based scheduler for computationally intensive image classification tasks executed by a fleet of off-grid, solar-powered nodes. The proposed approach maximizes the total number of processed images by autonomously controlling local processing rates and dynamically deciding whether each task should be executed immediately, stored in the local buffer, or offloaded to a neighbouring node.

To improve scalability and reduce communication overhead, we introduce a gossip-based communication strategy, referred to as "Gossip mode", in which agents exchange state information only with a small, randomly selected subset of neighbouring nodes. Unlike traditional centralized approaches, the proposed method requires neither global coordination nor prior knowledge of future solar energy production, enabling deployment in fully disconnected environments. The proposed scheduler is evaluated against an optimal Integer Linear Programming (ILP) benchmark and achieves performance close to the theoretical optimum, processing $96.8\%$ of the images under a fixed workload and $97.5\%$ under a variable acquisition rate, while successfully avoiding battery depletion.

# Reproducing the results
## Software requirments
All Python dependencies are listed in the `requirements.txt` file. The system has been tested with Python 3.12. We recommend creating a virtual environment to install and manage all required dependencies.
```bash
python3 -m venv env
source env/bin/active
pip install -r requirements.txt
```
## Integer Linear Programming (ILP) Method
The ILP formulation is used to obtain the optimal baseline for comparison with the proposed method. To execute the ILP solver, first navigate to the ILP folder and then run the corresponding Python script.
```bash
cd scripts/ilp
python main.py --days 7
```
While to run the version with a variable arrival rate:
```bash
python main.py --days 7 --variable-arrival-rate
```
The plots generated from the ILP experiments are available in the `ilp` folder.
## MADRL Method
To execute the MADRL scheduler, first navigate to the `nn` folder and then run the Python script using the following parameters:
```bash
cd scripts/nn
python -u sb3_main.py \
  --num-envs 15 \
  --num-agents 10 \
  --termination-mode early \
  --eval-termination-mode early \
  --battery-hard-threshold 0.2 \
  --seed random \
  --num-episodes 1000 \
  --net-width 256 \
  --net-layers 3 \
  --handshaking-weight 0 \
  --evaluation-interval 0 \
  --gamma 0.995 \
  --gossip \
  --offloading-weight 0.5 \
  --gossip-state-nodes 3 \
  --processed-images-weight 1 \
  --overflow-weight 2 \
  --gossip-order timestamp \
  --arrival-rate 20 \
  --battery-reward-weight 0
```
To run the variable-rate acquisition scenario, the `--variable-arrival-rate` flag must be added to the execution command:
```bash
cd scripts/nn
python -u sb3_main.py \
  --num-envs 15 \
  --num-agents 10 \
  --termination-mode early \
  --eval-termination-mode early \
  --battery-hard-threshold 0.2 \
  --seed random \
  --num-episodes 1000 \
  --net-width 256 \
  --net-layers 3 \
  --handshaking-weight 0 \
  --evaluation-interval 0 \
  --gamma 0.995 \
  --gossip \
  --offloading-weight 0.5 \
  --gossip-state-nodes 3 \
  --processed-images-weight 1 \
  --overflow-weight 2 \
  --gossip-order timestamp \
  --arrival-rate 20 \
  --battery-reward-weight 0 \
  --variable-arrival-rate
```
Then, a folder named `tb_logs` is generated, containing the TensorFlow logging data as well as the plots that describe the behavior of the nodes over time.