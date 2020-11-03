import Curriculum as gc 


class Memory(object):
    def __init__(self, steps_per_update, spatial_obs_shape, non_spatial_obs_shape, action_space):
        
        self.step = 0 
        self.max_steps = steps_per_update
        
        self.spatial_obs = torch.zeros(steps_per_update, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, 1)
        self.returns = torch.zeros(steps_per_update, 1)
        self.td_outcome = torch.zeros(steps_per_update, 1)
        
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, action_shape)
        self.actions = self.actions.long()
        self.action_masks = torch.zeros(steps_per_update, action_space, dtype=torch.uint8)

    def cuda(self):
        pass 
        
    def insert_worker_memory(self, worker_mem):
        
        steps_to_copy = worker_mem.get_steps_to_copy()
        
        # check that there's space left. 
        if self.max_steps - self.step < steps_to_copy: 
            steps_to_copy = self.max_steps - self.step 
        
        begin =  self.step
        end = self.step +  steps_to_copy
        
        self.spatial_obs[step:end].copy_(worker_mem.spatial_obs[:steps_to_copy]) 
        self.non_spatial_obs[step:end].copy_(worker_mem.non_spatial_obs[:steps_to_copy])  
        self.rewards[step:end].copy_(worker_mem.rewards[:steps_to_copy]) 
        self.returns[step:end].copy_(worker_mem.returns[:steps_to_copy]) 
        self.td_outcome[step:end].copy_(worker_mem.td_outcome[:steps_to_copy]) 
        
        self.actions[step:end].copy_(worker_mem.actions[:steps_to_copy]) 
        self.action_masks[step:end].copy_(worker_mem.action_masks[:steps_to_copy]) 

        self.step += steps_to_copy
    
    def not_full(self): 
        return 0.9*self.max_steps > self.steps 
    
class WorkerMemory(object): 
    def __init__(self, max_steps, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.max_steps = max_steps
        self.looped = False 
        self.step = 0 
        
        self.spatial_obs = torch.zeros(max_steps, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(max_steps, *non_spatial_obs_shape)
        self.rewards = torch.zeros(max_steps, 1)
        self.returns = torch.zeros(max_steps, 1)
        self.td_outcome = torch.zeros(max_steps, 1)
        
        action_shape = 1
        self.actions = torch.zeros(max_steps, action_shape)
        self.actions = self.actions.long()
        self.action_masks = torch.zeros(max_steps, action_space, dtype=torch.uint8)
 
    def cuda(self): 
        pass 
    
    def insert_network_step(self, spatial_obs, non_spatial_obs, action, reward, action_masks): 
        step = self.step 
        
        self.spatial_obs[step].copy_(spatial_obs)
        self.non_spatial_obs[step].copy_(spatial_obs)
        self.actions[step].copy_(action)
        self.reward[step].copy_(reward)
        self.action_masks[step].copy_(action_masks)
        
        self.step += 1 
        if self.step == self.max_steps: 
            self.step = 0 
            self.looped = True 
        
    def insert_scripted_step(self, spatial_obs, non_spatial_obs, reward): 
        # observation overwrites the previously inserted observations 
        self.spatial_obs[step].copy_(spatial_obs)
        self.non_spatial_obs[step].copy_(spatial_obs)
        
        # reward is added to the previously inserted reward 
        self.reward[step] += reward 
         
    def insert_epside_end(self, td_outcome): 
        
        self.td_outcome[:] = td_outcome
        
        # Compute returns 
        assert not (self.step == 0 and self.looped == False)
        if self.step != 0: 
            
            self.returns[ self.step - 1 ] = self.rewards[self.step -1 ]
            for i in reversed(range(self.step-1)):
                self.returns[i] = self.returns[step + 1] * gamma + self.rewards[i]
            
        if self.looped: 
            self.returns[-1] = gamma * self.returns[0] + self.rewards[-1] 
            for i in reversed(range(self.step+1 , self.max_steps-1)):
                self.returns[i] = self.returns[step + 1] * gamma + self.rewards[i]
            
         
    
    def insert_first_obs(self, spatial_obs, non_spatial_obs): 
        self.step = 0 
        self.looped = False 
        
        # Reset everything to zero to make sure. Remove when confirmed. 
        # TODO 
        
        
        self.spatial_obs[0].copy_(spatial_obs)
        self.non_spatial_obs[0].copy_(non_spatial_obs)
        
                  
class VecEnv():
    def __init__(self, envs, academy, starting_agent, ):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False
        self.academy = academy 
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        
        lectures = self.academy.get_lectures( len(self.remotes)) 
        for remote, lecture in zip(self.remotes, lectures:
            remote.send(('queue lecture', lecture))
            remote.send(('swap opp', starting_agent))
            remote.send(('swap trainee', starting_agent))
        
        steps_per_update = 2000
        self.memory = Memory(steps_per_update, spatial_obs_shape, (1, non_spatial_obs_shape), action_space)
        
    def step(self):
        
        while memory.not_full(): 
            for remote in self.remotes: 
                if remote.poll(): 
                    data = remote.recv()
                    self.memory.insert_data(data["memory"])
                    self.academy.report(data("lecture outcome")) 
                   
                    #TODO: queue another lecture? 
            
            sleep(0.01)
            
        return True 
    
    
    def update_trainee(self, agent): 
        for remote in self.remotes:
            remote.send(('swap trainee', agent))
        
    
    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)

def worker(remote, parent_remote, env, worker_id):
    parent_remote.close()

    trainee = None 
    lectures = [] 
    worker_running = True 
    initialized = False 
    
    with torch.no_grad(): 
        memory = WorkerMemory( """ TODO BLA BLA BLA """)
        
        lect = lectures.pop() 
        obs = env.reset(lect)
        
        spatial_obs, non_spatial_obs = trainee._update_obs(obs)
        memory.insert_first_obs(spatial_obs, non_spatial_obs)
        
        while worker_running:
            
            # Updates from master process? 
            while remote.poll() or not initialized:  
                command, data = remote.recv()
                if command == 'swap trainee': 
                    trainee = data
                elif command == 'queue lecture': 
                    lecture.append(data)
                elif command == 'close':
                    worker_running = False 
                    break 
                else: 
                    raise Exception("Unknown command to worker")
                    exit() 
                
                if not initialized: 
                    initialized = not (trainee is None or len(lectures)==0) 
            
            
            # Agent takes step 
            steps += 1
            data_from_agent =  trainee.act(game=None, env=env, obs=obs)  
            cnn_used_for_action = trainee.cnn_used_for_latest_action() 
            
            if cnn_used_for_action: 
                (action, actions, action_masks, value) = data_from_agent
                obs, reward, done, info = env.step(action)
                spatial_obs, non_spatial_obs = trainee._update_obs(obs)
                
                reward_shaped = reward_function(env, info, shaped=True)
                memory.insert_network_step(""" TODO BLA BLA """)
            else: 
                pos = data_from_agent.position 
                action  = {
                    'action-type': data_from_agent.action_type,
                    'x': None if pos is None else pos.x,
                    'y': None if pos is None else pos.y } 
                obs, reward, done, info, lect_outcome = env.step(action)
                spatial_obs, non_spatial_obs = trainee._update_obs(obs)
                
                reward_shaped = reward_function(env, info, shaped=True)
                memory.insert_scripted_step(""" TODO BLA BLA """)
            
            #Check progress and report back 
            if done: 
                
                td_outcome = 0.5*(1 + info['touchdowns'] - info['opp_touchdowns'])
                assert td_outcome in [0, 0.5, 1]
                assert type(lect_outcome) == gc.LectureOutcome 
                lect_outcome.steps = steps 
                
                memory.insert_epside_end( td_outcome=td_outcome ) 
                
                if len(lectures)>0: 
                    lect = lectures.pop() 
                
                obs = env.reset(lecture=lect)
                spatial_obs, non_spatial_obs = trainee._update_obs(obs)
                memory.insert_first_obs(spatial_obs, non_spatial_obs)
    
                steps = 0
                remote.send((memory, lect_outcome))
            
            if steps >= reset_steps:
                # If we  get stuck or something - reset the environment
                print("Max. number of steps exceeded! Consider increasing the number.")
            