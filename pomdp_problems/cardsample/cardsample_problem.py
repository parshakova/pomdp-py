"""

The classic Card Guessing problem.

n cards each with multiplicity m

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

States: at time t is a tuple of size 3*n, stores 
        counts of cards, actions and matches for each card of type k 
        [ (# t's: c_t=1, # t's: a_t=1, # t's: a_t=1=c_T), ..., (# t's: c_t=n, # t's: a_t=n, # t's: a_t=n=c_T)]
Actions: a in {1, 2, ..., n}
Rewards:
    +1 if a_t == c_t 
    0 otw
Observations: at time t is a tuple of size 1*n, stores 
        matches of actions and their mismatches for each card of type k 
        [(# t's: a_t=1=c_T, # t's: a_t=1!=c_T), ... , (# t's: a_t=n=c_T, # t's: a_t=n!=c_T)]

s1, b1, a1 -> s2, o2, r2, b2, a2 -> ...

s1 = counts for (c1)
s2 = counts for ((c1, a1), (c2))
.
.
.
s_{i+1} = counts for ((c1, a1), (c2, a2), ..., (ci, ai), (c_{i+1}))

s_t, a_t -> sample card c_tp1 -> s_tp1 = counts on s_t U {c_t, a_t} U {c_tp1}

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

def get_s_a_update(state, a):
    # 3*ct is already updated during t-1 step
    # update s state by recording matching info: action at == sp.card into sp.val
    sp_a = np.array(state.val)
    ct = state.card
    if a == ct:
        # update sp[3*ct + 1:3*ct + 3] += (1, 1) for match; 3*ct is already updated 
        sp_a[3*ct + 1: 3*ct + 3] += np.ones(2, np.int32)
    else:
        # update sp[3*a+1] for mismatch; counter for a
        sp_a[3*a + 1] += 1
    return sp_a



def sp_remove_card(next_state):
    # remove last sampled card, that has not been paired with the action yet
    val = np.array(next_state.val)
    if not next_state.terminal:
        val[3*next_state.card] -= 1
    return val


def create_all_states(state, states):
    if state.card == -1:
        # dummy state s_{-1}
        sp_a_val = np.array(state.val)
        for card in range(P.n):
            # update card c_{t+1} counter
            sp_a_val[3*card] += 1
            s_tp1 = State(tuple(sp_a_val), card)
            if s_tp1 not in states:
                states.add(s_tp1)
                #print(s_tp1)
                create_all_states(s_tp1, states)
            sp_a_val[3*card] -= 1
    else:
        for action in range(P.n):
            sp_a_val = get_s_a_update(state, action)
            deck = P.m - sp_a_val[0::3]
            if deck.sum() == 0:
                card = "$"
                terminal = True
                r = int(action == state.card)
                s_tp1 = State(tuple(sp_a_val), card, r=r, terminal=terminal)
                if s_tp1 not in states:
                    states.add(s_tp1)
                    #print(s_tp1)
            else:
                terminal = False
                available_cards = np.where(deck > 0)[0]
                for card in available_cards:
                    # update card c_{t+1} counter
                    sp_a_val[3*card] += 1
                    r = int(action == state.card)
                    s_tp1 = State(tuple(sp_a_val), card, r=r, terminal=terminal)
                    if s_tp1 not in states:
                        states.add(s_tp1)
                        #print(s_tp1)
                        create_all_states(s_tp1, states)
                    sp_a_val[3*card] -= 1
 

def create_all_observations(states):
    all_obs = set()
    for st in states:
        val = np.array(st.val)
        if val.sum() <= 1:
            # need to have s' <- s,a
            continue
        sp_a = sp_remove_card(st)
        a_matches = sp_a[2::3]
        a_mismatches = sp_a[1::3] - a_matches

        obs_val = np.zeros((2*P.n), np.int32)
        obs_val[::2] = a_matches
        obs_val[1::2] = a_mismatches
        obs = Observation(tuple(obs_val))
        if obs not in all_obs:
            #assert sp_a[1::3].sum() == obs_val.sum()
            #print(obs, st)
            all_obs.add(obs)
        
    return list(all_obs)


class State(pomdp_py.State):
    def __init__(self, val, card, r=None, terminal = False):
        """
        val: [ (# t's: c_t=1, # t's: a_t=1, # t's: a_t=1=c_T), ..., (# t's: c_t=n, # t's: a_t=n, # t's: a_t=n=c_T)]
        terminal: bool
        """
        self.val = val
        self.card = card
        self.r = r
        self.terminal = terminal

    def is_terminal(self):
        # |{c_t = k}|= m, forall k in [n]
        val = np.array(self.val)
        return (val[::3].sum() == P.n*P.m and val[1::3].sum() == P.n*P.m)

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
        return "State(%s, %s, %s)" % (str(self.val), str(self.card), str(self.r))

    def copy(self):
        return State(self.val, self.card, self.r, self.terminal)


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
        return "Action(%s)" % str(self.val)


class Observation(pomdp_py.Observation):
    def __init__(self, val):
        """
        val: [(# t's: a_t=1=c_T, # t's: a_t=1!=c_T), ... , (# t's: a_t=n=c_T, # t's: a_t=n!=c_T)]
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


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    # deterministic given s' and a

    def probability(self, observation, next_state, action):
        # Pr( o | s', a)
        # a = a_t
        # s' = [ (# t's: c_t=1, # t's: a_t=1, # t's: a_t=1=c_T), ..., (# t's: c_t=n, # t's: a_t=n, # t's: a_t=n=c_T)]
        # o = [(# t's: a_t=1=c_T, # t's: a_t=1!=c_T), ... , (# t's: a_t=n=c_T, # t's: a_t=n!=c_T)]
        # s_t, a_t -> sample card c_tp1 -> s_tp1 = counts on s_t U {c_t, a_t} U {c_tp1}

        obs_real = np.array(observation.val)
        sp_a = sp_remove_card(next_state)

        a_matches = sp_a[2::3]
        a_mismatches = sp_a[1::3] - a_matches

        assert np.all(a_mismatches >= 0), "prob false sp_a"
        assert np.all(sp_a >= 0), "prob false sp_a 2"

        if np.all(obs_real[::2] == a_matches) and np.all(obs_real[1::2] == a_mismatches):        
            return 1.
        else:
            return 0.


    def sample(self, next_state, action):
        # a = a_t
        # s' = [ (# t's: c_t=1, # t's: a_t=1, # t's: a_t=1=c_T), ..., (# t's: c_t=n, # t's: a_t=n, # t's: a_t=n=c_T)]
        # o = [(# t's: a_t=1=c_T, # t's: a_t=1!=c_T), ... , (# t's: a_t=n=c_T, # t's: a_t=n!=c_T)]
        # deterministic
        # s_t, a_t -> sample card c_tp1 -> s_tp1 = counts on s_t U {c_t, a_t} U {c_tp1}
        sp_a = sp_remove_card(next_state)
        a_matches = sp_a[2::3]
        a_mismatches = sp_a[1::3] - a_matches

        obs = np.zeros((2*P.n), np.int32)
        obs[::2] = a_matches
        obs[1::2] = a_mismatches
        try:
            assert np.all(a_mismatches >= 0)and np.all(sp_a >= 0), "prob false sp_a" 
        except:
            #print(next_state.val, next_state.card, sp_a, action, a_mismatches)
            assert 1/0
        return Observation(tuple(obs))

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return all_obs


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        # Pr( s' | s, a )
        # s' = [ (# t's: c_t=1, # t's: a_t=1, # t's: a_t=1=c_T), ..., (# t's: c_t=n, # t's: a_t=n, # t's: a_t=n=c_T)]
        # o = [(# t's: a_t=1=c_T, # t's: a_t=1!=c_T), ... , (# t's: a_t=n=c_T, # t's: a_t=n!=c_T)]
        # get last drawn card in s, c_t, and update counts in s.val for a current a
        # compare them to sp.val \ {c_{t+1}}
        # if  equal -> output the probability of sampling c_{t+1}
        # otherwise the probability is 0

        s_val = np.array(state.val)
        c_t = state.card
        c_tp1 = next_state.card
        at = action.val

        if state.terminal:
            return float(np.all(state.val == next_state.val) and next_state.terminal)

        sp_real_val = sp_remove_card(next_state)
        sp_a_val = get_s_a_update(state, action.val)

        assert np.all(sp_real_val >= 0), "prob false sp_a"

        if not (np.all(sp_a_val == sp_real_val)):
            return 0.

        if not next_state.terminal:
            # multiplicities of left over cards
            probs = P.m - s_val[::3]
            probs = probs*1.0/probs.sum()
            # return probability of sampling card c_{t+1}
            return probs[c_tp1]
        else:
            return 1.0 

    def sample(self, state, action):
        # s' ~ Pr( s' | s, a )
        # s = [ (# t's: c_t=1, # t's: a_t=1, # t's: a_t=1=c_T), ..., (# t's: c_t=n, # t's: a_t=n, # t's: a_t=n=c_T)]
        # Output:
        # s_t, a_t -> sample card c_tp1 -> s_tp1 = counts on s_t U {c_t, a_t} U {c_tp1}

        if state.terminal:
            return state.copy()

        sp_a_val = get_s_a_update(state, action.val)

        # multiplicities of left over cards
        probs = P.m - sp_a_val[::3]
        r = 0

        if probs.sum() == 0:
            card = "$"
            #print(state, action, sp_a_val)
            terminal = True
        else:
            probs = probs/probs.sum()
            card = np.random.choice(P.n, 1, replace=False, p=probs)[0]
            terminal = False
            # update card c_{t+1} counter
            sp_a_val[3*card] += 1
            r = int(action.val == state.card)

        return State(tuple(sp_a_val), card, r=r, terminal=terminal)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return all_states


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, next_state):
        if next_state.r == 1:
            return 1
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        """
        s_val = np.array(state.val)
        sp_val = np.array(next_state.val)
        sp_a_val = get_s_a_update(state, action.val)
        if not next_state.terminal:
            sp_val[3*next_state.card] -= 1
        if state.terminal:
            assert np.all(s_val == sp_val), "order of s and s' "
        elif not next_state.terminal:
            assert np.all(sp_a_val == sp_val), "order of s and s' %s %s %s %s %s %s" %\
                        (str(action), str(state), str(next_state), str(s_val), str(sp_a_val), str(sp_val)) 
                        """
        return self._reward_func(next_state)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is a random policy model To keep consistent
    with the framework."""
    ACTIONS = [Action(a) for a in range(P.n)]

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
        print("True state: %s, %d %d" % (true_state, np.array(true_state.val)[::3].sum(), int(true_state.terminal)))
        print("Action: %s" % str(action))
        print("Observation: %s,  %d" % (str(real_observation), np.array(real_observation.val).sum()))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))


def episode_planner(card_problem, planner, nsteps, reuse = False):
    """
    Runs the action-feedback loop of Card Guessing problem POMDP

    Args:
        card_problem (card_problem): an instance of the card guessing problem.
        planner: planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    s0 = env_reset_s0(card_problem)
    print("s0 ",   s0)
    total_reward = 0
    #policy = card_problem.agent.policy_model
    
    for i in range(nsteps):
        print("==== Step %d ====" % (i+1))
        if  reuse:
            action = max(card_problem.agent.tree.children, key=lambda a: card_problem.agent.tree.children[a].value)
        else:
            action = planner.plan(card_problem.agent)
        #action = policy.sample(card_problem.agent.cur_belief)

        true_state = copy.deepcopy(card_problem.env.state)
        env_reward = card_problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(card_problem.env.state)
        real_observation = card_problem.env.provide_observation(card_problem.agent.observation_model, action)
        card_problem.agent.update_history(action, real_observation)

        total_reward += np.maximum(0, env_reward)
        #print("belief ", card_problem.agent.cur_belief, len(card_problem.agent.cur_belief))
        print("True state: %s, %d %d" % (true_state, np.array(true_state.val)[::3].sum(), int(true_state.terminal)))
        print("Action: %s" % str(action))
        print("Observation: %s,  %d" % (str(real_observation), np.array(real_observation.val).sum()))
        print("Reward: %s" % str(np.maximum(0, env_reward)))
        print("True next state: %s, %d %d" % (true_next_state, 
            np.array(true_next_state.val)[::3].sum(), int(true_next_state.terminal)))
        

        print(card_problem.agent.tree, card_problem.agent.tree.children, card_problem.env.state, real_observation)
        vals = {a:card_problem.agent.tree.children[a].value for a in card_problem.agent.tree.children.keys()}
        print(vals)
        if i == 0:
            solved_tree = copy.deepcopy(card_problem.agent.tree)

        planner.update(card_problem.agent, action, real_observation)

        if isinstance(card_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(card_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          card_problem.agent.observation_model,
                                                          card_problem.agent.transition_model)
            card_problem.agent.set_belief(new_belief)

        

    print("Reward (Cumulative): %s" % str(total_reward))

    return total_reward, solved_tree




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

def init_belief_particle_s0(all_states):
    s0_list = []
    for s in all_states:
        if np.array(s.val).sum() == 1:
            s0_list += [s]

    init_belief = pomdp_py.Particles(s0_list)
    return init_belief

def init_belief_particle(all_states):
    init_belief = pomdp_py.Particles(all_states)
    return init_belief

def init_belief_histogram(all_states):
    # if we want to introduce the full belief space at once
    d = {}

    for s in all_states:
        if np.array(s.val).sum() == 1:
            d[s] = 1./P.n
        else:
            d[s] = 0.

    init_belief = pomdp_py.Histogram(d)
    return init_belief

def get_random_state():
    c_1 = random.choice(range(P.n))
    s_val = np.zeros(3*P.n, np.int32)
    s_val[3*c_1] += 1
    init_true_state = State(tuple(s_val), c_1)
    return init_true_state

all_states = set()
s0 = State(tuple(np.zeros(3*P.n, np.int32)), -1) 
create_all_states(s0, all_states)
all_states = list(all_states)
all_obs = create_all_observations(all_states)

print("|S| = %d" % len(all_states))
print("|O| = %d" %  len(all_obs))

def env_reset_s0(problem):
    s0 = get_random_state()
    problem.env.state.val = s0.val
    problem.env.state.card = s0.card
    problem.env.state.r = s0.r
    problem.env.state.terminal = False
    return s0

def mc_average(card_problem, planner, n_iter, file_name, init_b, reuse, T):
    rewards = np.zeros(n_iter)

    for it in range(n_iter):
        card_problem.agent.set_belief(init_b, prior = True)
        
        if it == 0 or not reuse:
            r, tree_i = episode_planner(card_problem, planner, nsteps=T,  reuse=False)
            
        else:
            r, _ =  episode_planner(card_problem, planner, nsteps=T, reuse = True)
        print(card_problem.agent.tree)

        card_problem.agent.tree = copy.deepcopy(tree_i)

        rewards[it] = r
        np.save(file_name, rewards)

        num = (rewards != 0).sum()

        print("average reward ", rewards.sum() * 1.0 / num, "*** iter = %d"%(it+1))

    print("average reward ", rewards.mean(), "over %d iter"%n_iter)

    return rewards

def main():

    T = P.m*P.n
    init_true_state= get_random_state()
    init_belief_hist = init_belief_histogram(all_states)
    init_belief_part = init_belief_particle_s0(all_states)
    card_problem = CardProblem(init_true_state, init_belief_hist)

    card_problem.agent.tree = None

    card_problem.agent.set_belief(init_belief_hist, prior=True)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(max_depth=T//2, discount_factor=0.95,
                           num_sims=40000, exploration_const=200,
                           rollout_policy=card_problem.agent.policy_model)

    n_iter = 100
    reuse = True
    uct_rewards = mc_average(card_problem, pouct, n_iter, "rewards_pouct%d.npy"%n_iter, init_belief_hist, reuse, T)



    print("*** Testing POMCP ***")

    card_problem.agent.tree = None
    n_iter = 100
    reuse = True

    pomcp = pomdp_py.POMCP(max_depth=T//2, discount_factor=1.,
                               num_sims=20000, exploration_const=200,
                               rollout_policy=card_problem.agent.policy_model,
                               num_visits_init=1)

    mcp_rewards = mc_average(card_problem, pomcp, n_iter, "rewards_pomcp%d.npy"%n_iter, init_belief_part, reuse, T)

    




if __name__ == '__main__':
    main()
