# Comparative study on the scalability of MARL for workload offloading in solar-powered edge networks

Engineering in Computer Science - Sapienza Università di Roma, A.Y. 2025/2026

Master's Thesis of Lorenzo Pecorari

#### Required libraries available in "requirements.txt"

 Single-agent RL:

    - position: ./scripts/qlt/no_drop
    - execution: python3 ./scripts/qlt/no_drop/qlt_main.py

 Multi-agent RL:

    - position (Tabular): ./scripts/multi-agent/custom-environment/mas_env/tabular-tests
    - execution: python3 ./scripts/multi-agent/custom-environment/mas_env/tabular-tests/mas_main.py

    - position (DQN): ./scripts/multi-agent/custom-environment/mas_env/nn
    - several observation designs for DQN approach, each has its own directory
    - execution: python3 ./scripts/multi-agent/custom-environment/mas_env/nn/<obs design>/sb3_main.py

#### Solar-irradiance data in ./dataset

#### Data for plots

Each configuration tested has its own directory, it may be needed to change the number of agents, the seed for the day and other customizable parameters. For comparing configurations with already existing data, inside the folder "data_and_plots" there exist a set of scripts allowing to generate most of interesting plots in the study