from numba import jit

@jit(nopython=True)
def jit_calculate_reward_local(
    fti,
    xti,
    gti,
    hti,
    ft_gti,
    xt_gti,
    gt_gti,
    ht_gti,
    irradiance,
    panel_area,
    panel_efficiency,
    backlog,
    backlog_gti,
    e_idle,
    e_frame,
    e_tx_rx,
    battery_level,
    battery_capacity,
    processing_rate,
    proc_interval,
    agent_id,
    ):
    panel_energy = irradiance * panel_area * panel_efficiency * proc_interval
    actual_battery = battery_level + panel_energy
    
    processable = max(min(backlog, int((actual_battery - e_idle) / e_frame), processing_rate * proc_interval), 0)    
    # processed = min(processable, fti * proc_interval)
    
    # needed_energy = processed * e_frame + e_idle
    # needed_energy = (fti * proc_interval * e_frame) + e_idle
    
    needed_energy = (fti * proc_interval * e_frame) + e_idle
        
    local_reward = 0
    
    if(actual_battery > needed_energy and processable > 0):
        processed = min(fti * proc_interval, processable)
        local_reward = (processed/processable) + (actual_battery/battery_capacity) + (processed / backlog)
        
        actual_battery = max(actual_battery - needed_energy, 0)
        backlog = max(backlog - processed, 0)
        # if(backlog > 0):
        #     local_reward = (processed/processable) * ((actual_battery - needed_energy)/battery_capacity) * 100 
        # else:
        #     local_reward = (processed / processable) * (actual_battery / battery_capacity) * processed
        
    else:
        local_reward = 0
        if(processable == 0 and fti == 0):
            local_reward = actual_battery / battery_capacity
            
        actual_battery = max(actual_battery - e_idle, 0)

    return local_reward

@jit(nopython=True)
def jit_calculate_reward_offloading(
    fti,
    xti,
    gti,
    hti,
    ft_gti,
    xt_gti,
    gt_gti,
    ht_gti,
    irradiance,
    panel_area,
    panel_efficiency,
    backlog,
    backlog_gti,
    e_idle,
    e_frame,
    e_tx_rx,
    battery_level,
    battery_capacity,
    processing_rate,
    proc_interval,
    arrival_rate,
    agent_id,
    w
    ):

    if(fti + hti) > processing_rate:
        hti = processing_rate - fti
                
    if(ft_gti + ht_gti) > processing_rate:
        ht_gti = processing_rate - ft_gti
            
    actual_battery = battery_level
    offloading_reward = 0
    remaining_framerate = processing_rate - fti
    
    if(remaining_framerate > 0 and xti != 0):
        
        if ft_gti + ht_gti > processing_rate:
            return 0
        
        if(xti == 1 # I want to send
            and gti != agent_id # Destination different than me
            and hti > 0 # Offloading framerate > 0
            and xt_gti == 2 # Destination in receive mode
            and gt_gti == agent_id # If target is me
            and ht_gti > 0 # Target has offloading framerate > 0
            and backlog_gti <= (proc_interval * arrival_rate)): # Backlog of target < new frames [NO SENSE]
            ht = min(hti, ht_gti)
            
            # needed_energy = ht * self._proc_interval * self.e_tx_rx
            processable = max(min(backlog, int((actual_battery - e_idle) / e_tx_rx), remaining_framerate * proc_interval), 0)
            # if(processable > 0):
            processed = min(ht * proc_interval, processable)
            needed_energy = ht * proc_interval * e_tx_rx
            
            if(needed_energy <= actual_battery and processable > 0):
                # actual_battery = max(actual_battery - needed_energy, 0)
                # backlog = max(backlog - processed, 0)
                offloading_reward = (processed/(processable)) + (actual_battery/battery_capacity) + (processed / backlog)

                # if(backlog > 0):
                #     offloading_reward = (processed/(processable)) * (actual_battery/battery_capacity) * (processed / backlog) 
                # else:
                    # offloading_reward = (processed/(processable)) * (actual_battery/battery_capacity) * processed 
            else:
                offloading_reward = 0
                if(processable == 0 and ht == 0):
                    offloading_reward = actual_battery / battery_capacity
        
        elif(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0 and backlog <= (proc_interval * arrival_rate)):
            ht = min(hti, ht_gti)
            # needed_energy = ht * self._proc_interval * self.e_tx_rx
            processable = max(min(backlog, int((actual_battery - e_idle) / (e_tx_rx + e_frame)), remaining_framerate * proc_interval), 0)
            # if(processable > 0):
            processed = min(ht * proc_interval, processable)
            needed_energy = ht * proc_interval * (e_tx_rx + e_frame)
            
            if(needed_energy <= actual_battery and processable > 0):
                offloading_reward = (processed/(processable)) + (actual_battery/battery_capacity)

            else:
                offloading_reward = 0
                if(processable == 0 and ht == 0):
                    offloading_reward = actual_battery / battery_capacity
                        
    return w * offloading_reward