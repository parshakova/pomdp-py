"""

The classic Card Guessing problem.

n cards each with multiplicity m

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

States: at time t its a tuple ((a_1 c_1) (a_2 c_2) ... (a_t c_t))
Actions: a in {1, 2, ..., n}
Rewards:
    +1 if a_t == c_t 
    0 otw
Observations: {a_i 1(a_i == c_i)}_{i in [t]}

"""

import pomdp_py
import random
import numpy as np
import sys
import heapq
import bisect
import copy

class P:
    m = 2
    n = 2
    size = len(str(n-1))
    count = 0

def create_all_states():
    output = []
    T = P.m*P.n
    t = 1; level_size= 0
    actions = [str(a) for a in range(P.n)]
    val_sets = set()

    deck = np.ones(P.n)*P.m
    for ci in range(P.n):
        deck[ci] -= 1
        output += [State(tuple([str(ci).zfill(P.size)]), tuple(deck))]
        val_sets.add(tuple([str(ci).zfill(P.size)]))
        deck[ci] += 1
    level_size = len(output)

    while t <= T:
        level = 0
        N = len(output)
        if t == T:
            available_cards = ["$"]
        for j in range(level_size):
            prev_s = output[N - level_size +j]
            deck = np.array(prev_s.deck)
            if t < T:
                available_cards = np.where(deck > 0)[0]
            for ci in available_cards:
                if t < T:
                    deck[ci] -= 1
                for ai in actions:
                    val = list(prev_s.val)
                    c_im1 = val.pop()
                    a_c_i = ai.zfill(P.size) + c_im1 # adjoint action-state info
                    bisect.insort(val, a_c_i)
                    val = val + [str(ci).zfill(P.size)] 
                    if tuple(val) in val_sets:
                        continue
                    val_sets.add(tuple(val)) 
                                      
                    output += [State(tuple(val), tuple(deck), ai == c_im1)]
                    level += 1
                if t < T:
                    deck[ci] += 1
        t += 1
        level_size = level
        
        
    return output


def create_all_observations(states):
    output = []
    val_sets = set()
    
    for s in states:
        val = s.val 
        if val in val_sets or len(val) == 1:
            continue
        val_sets.add(val)   
        oval = []         
        for at_ct in s.val[:-1]:
            at = at_ct[:P.size]
            ct = at_ct[-P.size:]
            oval +=  [at + str(int(at == ct))] 
        output += [Observation(tuple(oval))]
    return output


class State(pomdp_py.State):
    def __init__(self, val, deck, r=None):
        """
        val: always sorted list ["a1 c1", "a2 c2", ..., "at ct"]
        terminal: bool
        """
        self.val = val
        self.r = r
        self.deck = tuple(deck)
        self.terminal = val[-1] == "$"
        P.count += 1

    def __hash__(self):
        return hash(tuple(self.val))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.val == other.val
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        #return "State(%s, %s)" % (str(self.val), str(self.deck))
        return "State(%s)" % (str(self.val))

    def copy(self):
        return State(self.val, self.deck, self.r)


class State_2(pomdp_py.State):
    def __init__(self, val, r=None):
        """
        val: always sorted list ["a1 c1", "a2 c2", ..., "at ct"]
        terminal: bool
        """
        self.val = val
        self.r = r
        self.terminal = val[-1] == "$"
        self.count = P.count
        P.count += 1

    def get_deck(self):
        deck = np.ones(P.n)*P.m
        if len(self.val) == 1 and len(self.val[0]) == P.size:
            s_idx = int(val[0])
            deck[s_idx] -= 1
        else:
            for at_ct in self.val:
                deck[int(at_ct[-P.size:])] -= 1
        return deck

    def __hash__(self):
        return hash(tuple(self.val))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.val == other.val
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s, %d)" % (str(self.val), self.count)

    def copy(self):
        return State(self.val, self.deck, self.r)


class Action(pomdp_py.Action):
    def __init__(self, val):
        """
        val: in {0, 1, ..., n-1}
        """
        self.val = val

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.val == other.val
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Action(%s)" % self.val


class Observation(pomdp_py.Observation):
    def __init__(self, val):
        """
        val: list [(a1 1(c1==a1)), ..., ( at 1(ct==at))]
        """
        self.val = val

    def __hash__(self):
        return hash(tuple(self.val))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.val == other.val
        return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Observation(%s)" % str(self.val)

all_states = create_all_states()
all_obs = create_all_observations(all_states)



# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    # deterministic given s' and a

    def probability(self, observation, next_state, action):
        # Pr( o | s', a)
        # a = a_t
        # s' = ["a_i1 c_i1", ..., "a_itm1 c_itm1", "c_t"]
        # order in o may be differrent from s' in indicators
        # o = ["a_i1 ind_i1", ..., "a_it ind_it"] 
        # len(observation.val) + 1 == len(next_state.val), "Observation Model probability"
        # number of drawn cards
        T = len(observation.val)
        t = 0
        val = list(next_state.val)
        if len(observation.val) + 1 != len(val):
            return 0.
        while t < T:
            sp_a = val[t][:P.size]
            # count of states where (a_t=a,c_t!=a_t) and (a_t=a,c_t==a_t) 
            a_counts = [{'o':0, 'sp':0}, {'o':0, 'sp':0}]
            while t<T and val[t][:P.size] == sp_a:
                o_a = observation.val[t][:P.size]
                if o_a != sp_a:
                    return 0
                o_ind = int(observation.val[t][-1])
                a_counts[o_ind]['o'] += 1
                c_t = val[t][-P.size:]
                a_counts[sp_a == c_t]['sp'] += 1
                t += 1

            if a_counts[0]['o'] != a_counts[0]['sp'] or a_counts[1]['o'] != a_counts[1]['sp']:
                return 0. 
            
        return 1.

    def sample(self, next_state, action):
        # a = a_t
        # s' = ["a_i1 c_i1", ..., "a_it c_it", "c_tp1"]
        # o = ["a_i1 ind_i1", ..., "a_it ind_it"]
        # deterministic
        obs = [0]*(len(next_state.val) - 1)
        for t, at_ct in enumerate(next_state.val[:-1]):
            at = at_ct[:P.size]
            ct = at_ct[-P.size:]
            obs[t] = at.zfill(P.size) + str(int(at == ct))
        return Observation(obs)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return all_obs

# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        # Pr( s' | s, a )
        # see if rules are obeyed # cards is n, multiplicity m
        if len(next_state.val) -1 != len(state.val):
            return 0.
        c_tp1 = next_state.val[-1]
        val = list(state.val)
        c_tp = val.pop()
        at_ct = action.val.zfill(P.size) + c_tp # adjoint action-state info (at ct)
        bisect.insort(val, at_ct)
        if c_tp1 == "$":
            if val != next_state.val[:-1]:
                return 0.
            else:
                return 1.
                
        assert len(c_tp1) == P.size, "probability, last card storage mistake"
        probs = np.array(state.deck) 
        probs /= probs.sum()
        return probs[int(c_tp1)]


    def sample(self, state, action):
        # s' ~ Pr( s' | s, a )
        # s = ["a_i1 c_i1", ..., "a_it c_it", "c_tp1"]
        # Output:
        # s' = ["a_i1 c_i1",  ..., "a_it c_it", "a_itp1 c_itp1", "c_tp2"]

        if state.terminal:
            return state.copy()
        
        at = action.val
        deck = np.array(state.deck)
        val = list(state.val[:])
        ct = val.pop()
        
        assert len(val) + 1 == len(state.val), "error in copying the list"
        assert len(ct) == P.size, "last elem of val in state"
        at_ct = at.zfill(P.size) + ct # adjoint action-state info
        bisect.insort(val, at_ct)

        if len(val) == P.n*P.m:
            # transition to terminal state
            c_tp1 = "$"
        else:
            probs = deck / deck.sum()
            c_tp1 = str(np.random.choice(P.n, 1, replace=False, p=probs)[0]).zfill(P.size)
            # one card has been drawn from the deck 
            deck[int(c_tp1)] -= 1
            c_tp1 = c_tp1.zfill(P.size)

        val = val + [c_tp1]
        return State(tuple(val), tuple(deck), at == ct)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return all_states


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, next_state):
        if next_state.r:
            return 1
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        assert state.val[:-1] == tuple(sorted(state.val[:-1], reverse=False)), "order of s and s' "
        assert next_state.val[:-1] == tuple(sorted(next_state.val[:-1], reverse=False)), "order of s and s' "
        return self._reward_func(next_state)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is a random policy model To keep consistent
    with the framework."""
    ACTIONS = [Action(str(a).zfill(P.size)) for a in range(P.n)]

    def sample(self, state, **kwargs):
        return random.choice(self.get_all_actions())

    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS


class CardProblem(pomdp_py.POMDP):

    def __init__(self, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="CardProblem")


def test_formulation(card_problem, policy, nsteps):
    """
    Runs the action-feedback loop of Card Guessing problem POMDP

    Args:
        card_problem (card_problem): an instance of the card guessing problem.
        policy: policy
        nsteps (int): Maximum number of steps to run this loop.
    """
    total_reward = 0
    for i in range(nsteps):
        action = policy.sample(card_problem.agent.cur_belief)

        true_state = copy.deepcopy(card_problem.env.state)
        env_reward = card_problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(card_problem.env.state)

        real_observation = card_problem.env.provide_observation(card_problem.agent.observation_model,
                                                              action)
        card_problem.agent.update_history(action, real_observation)
        #planner.update(card_problem.agent, action, real_observation)
        total_reward += env_reward
        print("True state: %s, %d" % (true_state, len(true_state.val)))
        print("Action: %s" % str(action))
        print("Observation: %s,  %d" % (str(real_observation), len(real_observation.val)))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))


def test_planner(card_problem, planner, nsteps):
    """
    Runs the action-feedback loop of Card Guessing problem POMDP

    Args:
        card_problem (card_problem): an instance of the card guessing problem.
        planner: planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    total_reward = 0
    for i in range(nsteps):
        print("==== Step %d ====" % (i+1))
        action = planner.plan(card_problem.agent)

        true_state = copy.deepcopy(card_problem.env.state)
        env_reward = card_problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(card_problem.env.state)

        real_observation = card_problem.env.provide_observation(card_problem.agent.observation_model,
                                                              action)
        card_problem.agent.update_history(action, real_observation)
        planner.update(card_problem.agent, action, real_observation)
        total_reward += env_reward
        print("True state: %s, %d" % (true_state, len(true_state.val)))
        print("Action: %s" % str(action))
        print("Observation: %s,  %d" % (str(real_observation), len(real_observation.val)))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))

        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)

    return total_reward 



def init_belief_particle2():
    # if we dont want to introduce the full belief space
    # use Particle, to add new states into belief and not introduce them at once
    particles = [0]* P.n
    deck = np.ones(P.n)*P.m

    for i in range(P.n):
        deck[i] -= 1
        particles[i] = State(tuple([str(i).zfill(P.size)]), tuple(deck))
        deck[i] += 1

    init_belief = pomdp_py.Particles(particles)
    return init_belief

def init_belief_particle():
    init_belief = pomdp_py.Particles(all_states)
    return init_belief

def init_belief_histogram():
    # if we want to introduce the full belief space at once
    d = {}

    for i, s in enumerate(all_states):
        if i < P.n:
            d[s] = 1./P.n
        else:
            d[s] = 0.

    init_belief = pomdp_py.Histogram(d)
    return init_belief


def main():
    T = P.m*P.n+1
    deck = np.ones(P.n)*P.m
    s_idx = random.choice(range(P.n))
    deck[s_idx] -= 1
    init_true_state = State(tuple([str(s_idx).zfill(P.size)]), tuple(deck))
    init_belief = init_belief_particle()
    init_belief_hist = init_belief_histogram()
    card_problem = CardProblem(init_true_state, init_belief_hist)

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=T, discount_factor=1)
    test_planner(card_problem, vi, nsteps=T)

    print("** Testing Environment **")
    policy=card_problem.agent.policy_model
    test_formulation(card_problem, policy, nsteps=T)

    card_problem.agent.set_belief(init_belief)
    card_problem.agent.tree = None

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=6, discount_factor=1.,
                           num_sims=20000, exploration_const=20,
                           rollout_policy=card_problem.agent.policy_model,
                           num_visits_init=1)
    tt = test_planner(card_problem, pomcp, nsteps=T)



if __name__ == '__main__':
    main()
