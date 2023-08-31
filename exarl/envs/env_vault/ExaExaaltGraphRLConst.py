# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830

import numpy  as np
import gym
import os
from mpi4py import MPI
from exarl.utils.globals import ExaGlobals
from datetime import datetime

# run_name = 'Exaalt_GraphTD3-v3_ExaExaaltGraph-v3_AC_15nw_3sd_500NoS_150e_100eps_mask_PATCH_'
# run_name = "Exaalt_GraphTD3-v3_ExaExaaltGraph-v3_AC_250nw_50sd_100KNoS_125e_100eps_"

try:
    graph_size = ExaGlobals.lookup_params('graph_size')
except:
    graph_size = 10

# NAME = np.random.randint(99999)

run_name = str(ExaGlobals.lookup_params('experiment_id'))
now = datetime.now()
NAME = now.strftime("%d_%m_%Y_%H-%M-%S_")

run_name = NAME + run_name

def dirichlet_draw(alphas):
    sample = [np.random.gamma(a, 1) for a in alphas]
    sums   = sum(sample)
    sample = [x/sums for x in sample]
    return sample

def get_graph_dist(knownStates, state):
    all_keys   = [k for k,v in knownStates.items() if v != None]
    graph_dist = {}
    for loop_state in all_keys:
        graph_dist[loop_state] = 5
    graph_keys = [[state]]
    inc_keys   = [state]
    for ii in range(5):
        new_keys = []
        for key in graph_keys[ii]:
            graph_dist[key] = ii
            # tmp_keys = [x for x in knownStates[key].probs.keys() if x not in inc_keys]
            tmp_keys = [x for x in knownStates[key].counts.keys() if (x not in inc_keys) and (x in knownStates.keys()) and (knownStates[x] != None)]
            inc_keys += tmp_keys
            new_keys += tmp_keys
        graph_keys.append(new_keys)
            
    return graph_dist

def get_graph_adj(knownStates, state, database):
    adj_mat    = np.zeros([graph_size, graph_size*2])
    all_keys = sum(value != None for value in knownStates.values())
    # print("ALLKEYS: ", all_keys)

    # change to list knownStates.keys() where they are not equal to None
    graph_keys = [[state]]
    inc_keys   = [state]

    # Iterates 20 times
    for ii in range(graph_size):

        # Checks if # of included keys is equal to or greater than 20
        if len(inc_keys) >= graph_size:
            inc_keys = inc_keys[:graph_size]
            break

        if len(inc_keys) == all_keys:
            # Check to make sure tmp key is not equal to None
            tmp_keys = [x for x in knownStates.keys() if x not in inc_keys and knownStates[x] != None]
            if len(tmp_keys) > 0:
                if len(tmp_keys) + len(inc_keys) <= graph_size:
                    inc_keys += tmp_keys
                else:
                    inc_keys += tmp_keys[-(graph_size - len(inc_keys)):]
            break

        new_keys = []
        for key in graph_keys[ii]:
            tmp_keys = [x for x in knownStates[key].counts.keys() if (x not in inc_keys) and (x in knownStates.keys()) and (knownStates[x] != None)]
            
            inc_keys += tmp_keys
            new_keys += tmp_keys
            
        graph_keys.append(new_keys)
    
    # print("INC KEY: ", len(inc_keys))

    for ii, row_key in enumerate(inc_keys):
        # print("ROW KEY: ", row_key)
        for jj, col_key in enumerate(inc_keys):
            # print("COL KEY: ", col_key, " ", jj)
            if col_key in knownStates[row_key].counts.keys():
                adj_mat[ii,jj] = knownStates[row_key].counts[col_key]
                dat_pos = jj+graph_size
                adj_mat[ii,dat_pos] = database[row_key].count(col_key)
                # print("Adjacency Mat: ", adj_mat[ii,jj], " Database Mat: ", adj_mat[ii,dat_pos])
    # print("Adjacency Matrix: ", adj_mat)

    # adj_mat, dat_mat = np.split(adj_mat, 2, axis=1)

    return adj_mat, inc_keys

def VE(traj, knownStates, database, nWorkers, d_prior):
    print('running VE... '+str(len(knownStates.keys()))+' states discovered')
    knownStates_keys = [k for k,v in knownStates.items() if v != None]
    builds={}
    for a in knownStates_keys:
        builds[a]=0

    taskList=[]
    for _ in range(nWorkers):
        virtuallyConsumed={}
        for i in knownStates_keys:
            virtuallyConsumed[i]=0

        state=traj[-1]
        while(len(database[state])+builds[state]>virtuallyConsumed[state]):
            virtuallyConsumed[state]+=1
            #print(database[state])
            try:
                state=database[state][virtuallyConsumed[state]-1]
                #print('virtual consumed to state ',state)
            except:
                # offset      = np.array([20., 0., 0., 0., 0., 0.])
                offset      = np.array([0., 0., 0., 0., 0., 0.])
                prior_p     = d_prior[0] + offset + 1.e-6
                graph_dist  = get_graph_dist(knownStates, state)
                d_alpha     = np.array([ prior_p[graph_dist[key]] for key in knownStates_keys ])
                # print("dalpha: ", d_alpha)
                # print("KNOWN STATES: ", knownStates.keys(), d_alpha)
                # print(graph_dist)
                # print("Dist 1:", knownStates[state].probs.keys())
                # print(d_prior)
                keylist = list(knownStates_keys)
                for ii in range(d_alpha.size):
                    try:
                        d_alpha[ii] = d_alpha[ii] + knownStates[state].counts[keylist[ii]]
                    except:
                        pass
                # count_num = np.array([knownStates[state].counts[x] if x in knownStates[state].counts.keys() else 0 for x in knownStates.keys()])
                if len(d_alpha) == 1:
                    sample_p = [1.]
                else:
                    #print("count: ", count_num)
                    #print("d_alpha: ", d_alpha)
                    # sample_p   = np.random.dirichlet(count_num+d_alpha)
                    # sample_p   = np.random.dirichlet(d_alpha)
                    sample_p   = dirichlet_draw(d_alpha)
                    #print("sample_p: ", sample_p)
                # print(count_num)
                # print(sample_p)
                state      = np.random.choice(list(knownStates_keys),p=sample_p)
                # print("STATE: ",state)
                # print("KNOWNSTATE KEYS: ", knownStates[state].probs.keys())
                # print("====================")
                #print('VE SAMPLE FOR PENDING SEGMENT ->',state)
        #print('scheduling in state ',state)
        taskList.append(state)
        builds[state]+=1
    return taskList

class StateStatistics:
    #constructor to initialize StateStatistic object
    def __init__(self, label, Map):
        self.Map    = Map
        self.counts = {}
        for neigh in self.Map[int(label)].keys():
            self.counts[neigh]=0
        #print('state '+str(label)+' will connect to: '+str(self.counts.keys()))
        self.nSegments=0
        self.nTransitions=0
        self.label = int(label)
        self.probs={}
        self.l=0
        self.lp=0
        self.initP=-1

    #clear old probabilities
    def clearProbs(self):
        self.probs={}

    #update state statistics
    def update(self, finalState):
        self.nSegments+=1
        try:
                self.counts[finalState]+=1
        except:
                self.counts[finalState]=1
        if(finalState!=self.label):
                self.nTransitions+=11

class ExaExaaltGraphRLConst(gym.Env):

    metadata = {"node_count": 2500}

    def __init__(self,**kwargs):
        super().__init__()
        """
        """
        stateDepth       = 25 #segments
        number_of_states = 2500

        self.n_states  = number_of_states
        self.graph_len = graph_size
        self.nWorkers  = 25
        self.reward    = 0
        self.WCT       = 0
        self.RUN_TIME  = int(ExaGlobals.lookup_params('n_steps'))
        
        self.database    = {}
        self.knownStates = {}
        self.selfTrans   = []
        self.selfTrans.append(StateStatistics(0, [{0:0,1:1}]))
        self.actions_avail = np.arange(0, self.n_states, 1)
        self.traj        = []

        self.state_order = [ii for ii in range(self.n_states)]

        self.Map={}
        side= 100
        for i in range(number_of_states):
            self.Map[i]={}
            self.Map[i][i]=1-1.0/stateDepth
        
            R=(1-self.Map[i][i])/4
            if(i%side==0):
                #left side
                self.Map[i][(i-1+side)%number_of_states]=R
                self.Map[i][(i+1)%number_of_states]=R
            else:
                if(i%side==(side-1)):
                    #right side
                    self.Map[i][i-1]=R
                    self.Map[i][(i+1-side)%number_of_states]=R
        
                else:
                    #not an edge
                    self.Map[i][i-1]=R
                    self.Map[i][(i+1)%number_of_states]=R
        
            self.Map[i][(i-side)%number_of_states]=R
            self.Map[i][(i+side)%number_of_states]=R

        self.INITIAL_STATE              = int(((side/2)*side+side/2)/100) 
        # self.INITIAL_STATE = np.random.randint(0,self.n_states)   
        # print("INITIAL STATE: ", type(self.INITIAL_STATE))
        # print("MAP at init: ", self.Map[self.INITIAL_STATE])



        self.database[self.INITIAL_STATE]    = []
        self.traj.append(self.INITIAL_STATE)    

        self.adj_states = np.zeros(self.graph_len, dtype=int)
        self.adj_states[0] = self.INITIAL_STATE                       


        # Set bounds using the structure in the Cartpole example
        # high = np.repeat(np.finfo(np.float32).max, self.n_obs)

        # self.action_space      = gym.spaces.Box(np.zeros(self.n_states), np.ones(self.n_states))
        # self.observation_space = gym.spaces.Box(np.zeros(self.n_states), np.ones(self.n_states))

        # Action space is going to change to represent the number actions queued for trajectories (See the VE algorithm nWorkers tasklist for size model)
        # self.action_space      = gym.spaces.Box(np.zeros(6), np.array([100.,100.,100.,100.,100.,100.]))
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.graph_len,), dtype=np.float32)

        # Adj matrix for the NN
        self.adj_obs_space = gym.spaces.Box(low=0, high=np.inf, shape=(graph_size, graph_size*2))
        
        # Position of the final trajectory
        self.end_traj = gym.spaces.Discrete(self.nWorkers)
        
        # Creation of the mask for the invalid action masking
        # MAY NEED TO CHANGE IF VALUE GETS TOO HIGH!!!
        self.mask = gym.spaces.Dict({})
        for i in range(number_of_states):
            self.mask[i] = gym.spaces.Discrete(65535)
            self.knownStates[i] = None

            # self.mask.update({i,gym.spaces.Discrete(65535)})
        
        self.knownStates[self.INITIAL_STATE] = StateStatistics(self.INITIAL_STATE, self.Map) 
        # print("Known States INITIAL STATE: ", self.knownStates[self.INITIAL_STATE])
        # print("Known States RANDOM STATE: ", self.knownStates[9100])


        # print(len(self.mask))

        # self.end_traj = gym.spaces.Discrete(self.n_states)
        self.observation_space = gym.spaces.Tuple((self.adj_obs_space, self.end_traj, self.mask))


        # Original observation space
        # self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(graph_size, graph_size))

    def crankModel(self):
        l={}
        pi={}
        for k in self.knownStates.keys():
            l[k]=0
            pi[k]=1
        for i in self.knownStates.keys():
            for j in self.Map[i].keys():
                if(j in self.knownStates.keys()):
                    l[i]+=0.5*(self.knownStates[i].counts[j]+self.knownStates[j].counts[i])
    
        iterations=0
        while(True):
            iterations+=1
            nextl={}
            for k in self.knownStates.keys():
                    nextl[k]=0
            for i in self.knownStates.keys():
                for j in self.Map[i].keys():
                    if(j in self.knownStates.keys()):
                        if(self.knownStates[i].counts[j]+self.knownStates[j].counts[i]>0):
                            nextl[i]+=((self.knownStates[i].counts[j]+self.knownStates[j].counts[i])*l[i]*pi[j])/(l[j]*pi[i]+l[i]*pi[j])
            #check accuracy
            accuracy=0
            for i in self.knownStates.keys():
                    accuracy+=np.abs(l[i]-nextl[i])
            if(accuracy<1e-7 or iterations>9):
                #print (accuracy)
                #transfer lambdas
                for i in self.knownStates.keys():
                    l[i]=nextl[i]
                break
            else:
                #print (accuracy)
                #transfer lambdas
                for i in self.knownStates.keys():
                    l[i]=nextl[i]
    
        for i in self.knownStates.keys():
            self.knownStates[i].clearProbs()
            for j in self.knownStates[i].counts.keys():
                if(j in list(self.knownStates.keys())):
                    if(self.knownStates[i].counts[j]+self.knownStates[j].counts[i]>0 and i!=j):
                        self.knownStates[i].probs[j]=((self.knownStates[i].counts[j]+self.knownStates[j].counts[i])*pi[j])/(l[j]*pi[i]+l[i]*pi[j])
            if(sum(self.knownStates[i].probs.values())<1):
                self.knownStates[i].probs[i]=1-sum(self.knownStates[i].probs.values())
            if(sum(self.knownStates[i].probs.values())>1):
                norm=sum(self.knownStates[i].probs.values())
                for key in self.knownStates[i].probs.keys():
                    self.knownStates[i].probs[key]=self.knownStates[i].probs[key]/norm
            """
            for neighborState in self.knownStates[i].counts.keys():
                try:
                    print('probs ',self.knownStates[i].label,'->',neighborState,self.knownStates[i].probs[neighborState])
                except:
                    0
            """

    def schedule(self, WCT):
        self.crankModel()
        taskList=VE(self.traj, self.knownStates, self.database, self.nWorkers)
        for i in range(self.nWorkers):
            workerID=i
            buildState=taskList[i]
            endState=np.random.choice(list(self.Map[buildState].keys()),p=list(self.Map[buildState].values()))
            self.knownStates[buildState].update(endState)
            self.database[buildState].append(endState)
            if(endState not in self.knownStates.keys()):
                self.knownStates[endState]=StateStatistics(endState,self.Map)
                self.database[endState]=[]
            #print(WCT,workerID,buildState,endState)
            # with open("traceOutput_2dModel"+str(NAME), "a") as myfile:
            #     myfile.write(
            #                 str(round(WCT,3))+' '+
            #                 str(workerID)+' '+
            #                 str(buildState)+' '+
            #                 str(endState)+' '+
            #                 '\n')
        pass

    def step(self, action):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        done = False
        # launchStates = np.random.choice(range(self.n_states), size=self.nWorkers)#, p=action)
        # self.crankModel()
        # print("Action: ", action)
        # Replace the task list with what the RL algorithm outputs for it's actions
        # This is where the VE algorithm is making the decisions of the actions to take
        # The task list is a list of indices that are length of nWorkers that represent the desired trajectories to be calculated
        
        # task_list = np.random.choice(self.actions_avail, size=self.num_actions, replace=True, p=sampled_actions_probs)
        # all_keys = [k for k,v in self.knownStates.items() if v != None]
        # print("Allowed keys: ", all_keys)

        # summation = 0
        # for i in range(len(all_keys)):
        #     print("Prob of hitting key ", all_keys[i], " = ", action[all_keys[i]])
        #     summation += action[all_keys[i]]

        # print("Sumamtion: ", summation)

        # print("Action: ", action)
        
        taskList = np.random.choice(self.adj_states, size=self.nWorkers, replace=True, p=action)
        # taskList = np.append(taskList, self.traj[-1])

        # taskList = VE(self.traj, self.knownStates, self.database, self.nWorkers, action)
        # print("End Traj: ", self.traj[-1])
        # print("Task List: ", taskList)

        # taskList = [self.INITIAL_STATE] * self.nWorkers

        # print("Tasklist: ", taskList)
        for i in range(self.nWorkers):
            workerID   = i
            buildState = taskList[i] # launchStates[i]
            # End state is the determined trajectory of the starting state set in the task list
            # The probability of transition is based of the map and is not random but set specifically at the beginning
            endState   = np.random.choice(list(self.Map[buildState].keys()),p=list(self.Map[buildState].values()))

            self.knownStates[buildState].update(endState)
            # if buildState == endState:
            #     self.selfTrans[0].update(0)
            # else:
            #     self.selfTrans[0].update(1)
            self.database[buildState].append(endState)
            # New end state that has not been transitioned to before needs to be addede to knownStates along with added into a database entry
            # self.knownStates = None
            if (self.knownStates[endState] == None):
            # if (endState not in self.knownStates.keys()):
                self.knownStates[endState]=StateStatistics(endState, self.Map)
                self.database[endState] = []
            # with open("traceOutput_2dModel"+str(NAME), "a") as myfile:
            #     myfile.write(
            #             str(round(self.WCT, 3)) + " "+
            #             str(workerID) + " " +
            #             str(buildState) + " " +
            #             str(endState) +" " +
            #             '\n')
        added = 0
        self.WCT+=1
        action_str = [str(x) for x in action]
        action_str = " ".join(action_str)
        taskList = list(taskList)
        # print("Untouched tasklist: ", taskList)

        if not os.path.exists("./outputs/Exaalt/" + run_name + "/"):
            os.makedirs("./outputs/Exaalt/" + run_name + "/")

        while(True):
            current_state=self.traj[-1]
            next_state = None
            try:
                next_state=self.database[current_state].pop(0)
                # print("Next state: ", next_state)
                self.traj.append(next_state)
            except:
                with open("./outputs/Exaalt/" + run_name + "/" + run_name + "_" + str(rank), "a") as myfile:
                        myfile.write(
                        str(round(self.WCT,3))+' '+
                        str(len(self.traj))+' '+
                        str(self.WCT*self.nWorkers)+' '+
                        str((len(self.traj)-1)/float(self.WCT*self.nWorkers))+' '+
                        str(added)+' '+
                        action_str + ' ' +
                        '\n')
                break
            if (next_state in taskList):
                added += 1
                taskList.remove(next_state)
                # print("Modified tasklist: ", taskList)
                
        # self.reward =  ((self.RUN_TIME-self.WCT)/self.RUN_TIME)*(added/self.nWorkers)
        
        self.reward = added/self.nWorkers

        # if (self.WCT >= self.RUN_TIME):
        #     self.reward = (len(self.traj)-1)/float(self.WCT*self.nWorkers) 
        #     done = True

        """ Iterates the testing process forward one step """

        # self.reward = 0.5*(len(self.traj)-1)/float(self.WCT*self.nWorkers) + 0.5*(added/self.nWorkers)
        # self.reward = action[1]
        # self.reward = (len(self.traj)-1)/float(self.WCT*self.nWorkers)
        current_state = self.traj[-1]
        adj_mat, inc_keys = self.generate_data()
        np.put(self.adj_states, np.arange(len(inc_keys)), inc_keys)        
        next_state = (adj_mat, current_state, self.knownStates)

        info = None
        
        print("Step: ", self.WCT, " Reward: ", self.reward, " ", done, " Added: ", added)
        return next_state, self.reward, done, info

    def reset(self):
        """ Start environment over """
        side = 100

        self.WCT                             = 0 
        self.INITIAL_STATE                   = int(((side/2)*side+side/2)/100)
        # self.INITIAL_STATE = np.random.randint(0,self.n_states)  
        self.reward                          = 0 
        self.traj                            = []
        self.database                        = {}
        self.selfTrans                       = []
        self.selfTrans.append(StateStatistics(0, [{0:0,1:1}]))
        self.knownStates                     = {}
        self.database[self.INITIAL_STATE]    = []
        self.traj.append(self.INITIAL_STATE)
        self.adj_states = np.zeros(self.graph_len, dtype=int)
        self.adj_states[0] = self.INITIAL_STATE      

        for i in range(self.n_states):
            self.knownStates[i] = None
        
        self.knownStates[self.INITIAL_STATE] = StateStatistics(self.INITIAL_STATE, self.Map)
        adj_mat, inc_keys = self.generate_data()

        state_tuple = (adj_mat, self.traj[-1], self.knownStates)
        return state_tuple, {} # Return new state

    def render(self):
        """ Not relevant here but left for template convenience """
        return 0
    
    def generate_data(self):
        return get_graph_adj(self.knownStates, self.traj[-1], self.database)

