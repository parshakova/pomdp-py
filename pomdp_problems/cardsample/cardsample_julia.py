# Tiger POMDP from Kaelbling et al. 98 (http://www.sciencedirect.com/science/article/pii/S000437029800023X)
#from julia.api import Julia
#jl = Julia(compiled_modules=False)
import random
import numpy as np

from julia.QuickPOMDPs import *
from julia.POMDPs import solve, pdf
from julia.QMDP import QMDPSolver
from julia.SARSOP import SARSOPSolver
from julia.POMDPSimulators import stepthrough
from julia.POMDPPolicies import alphavectors

class P:
    m = 1
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


class State:
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


class Action:
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


class Observation:
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


all_states = set()
s0 = State(tuple(np.zeros(3*P.n, np.int32)), -1) 
create_all_states(s0, all_states)
all_states = list(all_states)
all_obs = create_all_observations(all_states)

print("|S| = %d" % len(all_states))
print("|O| = %d" %  len(all_obs))

S = all_states
A = [Action(a) for a in range(P.n)]
O = all_obs
γ = 1

def T(state, action, next_state):
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


def Z(action, next_state, observation):
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

def R(state, action):
    if state.card == action.val:
        return 1
    else: # listen
        return -1

m = DiscreteExplicitPOMDP(S,A,O,T,Z,R,γ)

solver = SARSOPSolver()
policy = solve(solver, m)

print('alpha vectors:')
for v in alphavectors(policy):
    print(v)

print()

rsum = 0.0
for step in stepthrough(m, policy, max_steps=10):
    print('s:', step.s)
    print('b:', [pdf(step.b, x) for x in S])
    print('a:', step.a)
    print('o:', step.o, '\n')
    rsum += step.r

print('Undiscounted reward was', rsum)
