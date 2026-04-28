import interpol as ip
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

EPS = 0.5
class SB3_MAS_Train:
    def __init__(self, 
            num_agents: int, 
            irradiance_datapaths: list,
            delta_time: int,
            proc_interval: int,
            proc_rate: int,
            arrival_rate: int,
            eps_init: float,
            eps_fin: float,
            eps_dec: float,
            battery_capacities: list,
            panel_surfaces: list,
            power_idle: float,
            power_max: float,
            w: float):
    
        self.num_agents = num_agents
        self.irradiance_datapaths = irradiance_datapaths
        self.delta_time = delta_time
        self.proc_interval_s = proc_interval
        self.proc_rate = proc_rate
        self.arrival_rate = arrival_rate
        self.eps_init = eps_init
        self.eps_fin = eps_fin
        self.eps_dec = eps_dec
        self.battery_capacities_wh = battery_capacities #Wh
        self.panel_surfaces = panel_surfaces
        self.power_idle = power_idle
        self.power_max = power_max
        self.w = w

        self.e_idle = power_idle * self.proc_interval_s
        self.e_frame = (0.8 * (power_max - power_idle) * 1) / proc_rate
        self.e_tx_rx = (0.2 * (power_max - power_idle) * 1) / proc_rate

        self.irradiance_data = []
        self.irradiance_arrays = []
        # Battery starts at 50% of its capacity
        day = 355
        for filepath in self.irradiance_datapaths:
            # print(filepath, delta_time, proc_interval)
            df = ip.interpolate(filepath, delta_time, proc_interval)
            self.irradiance_data.append(df)
            self.irradiance_arrays.append(df['ghi'].values[day*24*60//5:(day+1)*24*60//5])

        N = self.num_agents
        T = len(self.irradiance_arrays[0])
        print("N:", N, "T:", T)
        A = np.full((N, T), self.arrival_rate * self.proc_interval_s)
        x = cp.Variable((N, N, T), nonneg=True, integer=True)
        B = cp.Variable((N, T+1), nonneg=True)
        E = cp.Variable((N, T+1))
        spill = cp.Variable((N, T), nonneg=True)       
        # SLACK REMOVED — forces strict E >= 0 like RL environment
        # slack = cp.Variable((N, T), nonneg=True)
        y = cp.Variable((N, N, T), boolean=True) # Unicast transmission indicator
        
        self.Eharv = np.array(self.irradiance_arrays) # (N, T)
        self.Eharv = np.array([e[:T] for e in self.Eharv])

        constraints = []

        # Backlog starts at 0
        B0 = np.zeros(N)
        constraints += [B[:, 0] == B0]

        # Keep energy units consistent in Joules across the model.
        E0_vec = np.array(self.battery_capacities_wh) * 0.5 * 3600
        constraints += [E[:, 0] == E0_vec]

        constraints += [E >= 0]
        constraints += [E <= np.array(self.battery_capacities_wh).reshape(-1, 1)*3600]

        # Set upper bound for x
        UB = self.proc_rate * self.proc_interval_s
        constraints += [x <= UB]

        for t in range(T):
            out_i = []
            in_i = []
            for i in range(N):
                out_i.append(cp.sum(x[i, :, t]) - x[i, i, t])  # sum_{j!=i} x_ij
                in_i.append(cp.sum(x[:, i, t]) - x[i, i, t])   # sum_{h!=i} x_hi

            for i in range(N):
                # Unicast transmission constraint: an agent can only send to ONE other agent at time t
                constraints += [ cp.sum(y[i, :, t]) - y[i, i, t] <= 1 ]
                
                # Big-M constraint to link x and y for transmissions
                for j in range(N):
                    if i != j:
                        constraints += [ x[i, j, t] <= UB * y[i, j, t] ]

                # same-slot availability
                constraints += [
                    x[i, i, t] + out_i[i] <= B[i, t] + A[i, t] + in_i[i]
                ]

                # backlog conservation
                constraints += [
                    B[i, t+1] == B[i, t] + A[i, t] + in_i[i] - out_i[i] - x[i, i, t]
                ]

                # CPU cap
                constraints += [
                    x[i, i, t] <= self.proc_rate * self.proc_interval_s
                ]

                panel_energy_j = self.Eharv[i, t] * self.proc_interval_s * self.panel_surfaces[i] * 0.2
                # Energy conservation WITHOUT slack — strict E >= 0
                constraints += [
                    E[i, t+1] == E[i, t] + panel_energy_j - self.e_idle
                    - self.e_frame * x[i, i, t]
                    - self.e_tx_rx * (out_i[i] + in_i[i])
                    - spill[i, t]
                ]

        processed_total = cp.sum([x[i, i, k] for i in range(N) for k in range(T)])
        tx_total = cp.sum([x[i, j, k] for i in range(N) for j in range(N) if j != i for k in range(T)])

        SPILL_PENALTY = 1e-3
        EPSILON_OBJ = 1e-6

        objective = cp.Maximize(
            processed_total
            - EPSILON_OBJ * tx_total
            - SPILL_PENALTY * cp.sum(spill)
        )
        self.x = x
        self.prob = cp.Problem(objective, constraints)
        self.battery = E
        self.buffer = B
        self.slack = None  # Slack removed

    def solve(self):
        self.prob.solve(verbose=True,
            solver=cp.HIGHS,
            time_limit=60,
            random_seed=1234,
            threads=1)
        
        print("Status:", self.prob.status)
        print("Optimal value:", self.prob.value)
        print("Processed total:", sum([self.x[i, i, k].value for i in range(self.num_agents) for k in range(len(self.irradiance_arrays[0]))]))
        print("Tx total:", sum([self.x[i, j, k].value for i in range(self.num_agents) for j in range(self.num_agents) if j != i for k in range(len(self.irradiance_arrays[0]))]))

    def print_solution(self):
        if self.prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            print(f"No plottable solution (status: {self.prob.status}).")
            return

        plt.figure(figsize=(12, 24))
        for agent_id in range(self.num_agents):
            plt.subplot(self.num_agents, 1, agent_id + 1)

            ax1 = plt.gca()
            ax1.plot(self.x[agent_id, agent_id, :].value / (self.proc_rate * self.proc_interval_s), label='Processed Locally', color='tab:blue')
            
            irradiance = self.Eharv[agent_id, :] * self.proc_interval_s * self.panel_surfaces[agent_id] * 0.2
            max_irradiance = 1000 * self.proc_interval_s * self.panel_surfaces[agent_id] * 0.2
            ax1.plot(irradiance * 2 / max_irradiance, label='Irradiance', color='tab:orange')
            
            ax1.plot(
                self.battery[agent_id, :].value / (self.battery_capacities_wh[agent_id] * 3600),
                label='Battery Level', color='tab:red'
            )
            ax1.plot(self.buffer[agent_id, :].value / (self.arrival_rate * self.proc_interval_s * 150), label='Buffer Level', color='tab:purple')

            # Tx frames to nodes different from itself
            tx_total = np.array([ np.sum(self.x[agent_id, :, t].value) - self.x[agent_id, agent_id, t].value for t in range(len(self.irradiance_arrays[0]))])
            ax1.plot(tx_total / (self.proc_rate * self.proc_interval_s), label='Frames Transmitted', color='tab:green')
            
            # Rx frames from nodes different from itself
            rx_total = np.array([ np.sum(self.x[:, agent_id, t].value) - self.x[agent_id, agent_id, t].value for t in range(len(self.irradiance_arrays[0]))])
            ax1.plot(rx_total / (self.proc_rate * self.proc_interval_s * self.num_agents), label='Frames Received', color='tab:cyan')
            
            # Limita l'asse Y per evitare che i picchi enormi schiaccino la linea a 0.75
            ax1.set_ylim(-0.05, 1.05)
            
            plt.title(f'Agent {agent_id}')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.savefig('ilp_solution.png')
        #plt.show()

