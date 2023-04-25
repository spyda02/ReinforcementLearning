# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            newvals = util.Counter()
            for st in self.mdp.getStates():
                vals = []
                for act in self.mdp.getPossibleActions(st):
                    vals.append(self.computeQValueFromValues(st, act))
                if vals:
                    newvals[st] = max(vals)
            self.values = newvals


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qval = 0
        for next, p in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += p * (self.mdp.getReward(state, action, next) + self.discount * self.values[next])
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestact = None
        bestval = float('-inf')
        for act in self.mdp.getPossibleActions(state):
            qv = self.computeQValueFromValues(state, act)
            if qv > bestval:
                bestval = qv
                bestact = act
        return bestact

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        all_s = self.mdp.getStates()
        length = len(all_s)
        for i in range(self.iterations):
            s = all_s[i % length]
            if self.mdp.isTerminal(s) == False:
                bestaction = []
                for act in self.mdp.getPossibleActions(s):
                    val = 0
                    for nact, p in self.mdp.getTransitionStatesAndProbs(s, act):
                        val += p * (self.mdp.getReward(s, act, nact) + self.discount * self.values[nact])
                    bestaction.append(val)
                self.values[s] = max(bestaction)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***" 

        predecessors = {}
        pq = util.PriorityQueue()

        for st in self.mdp.getStates():
            if self.mdp.isTerminal(st):
                continue
            else:
                for act in self.mdp.getPossibleActions(st):
                    for nst, _ in self.mdp.getTransitionStatesAndProbs(st, act):
                        if nst in predecessors:
                            predecessors[nst].add(st)
                        else:
                            predecessors[nst] = {st}
        
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            else:
                diff = abs(self.values[s] - self.computeQValueFromValues(s, self.computeActionFromValues(s)))
                pq.push(s, -diff)
        
        for _ in range(self.iterations):
            if pq.isEmpty():
                break

            val = pq.pop()
            self.values[val] = self.computeQValueFromValues(val, self.computeActionFromValues(val))

            for temp in predecessors[val]:
                diff = abs(self.values[temp] - self.computeQValueFromValues(temp, self.computeActionFromValues(temp)))
                if diff > self.theta:
                    pq.update(temp, -diff)


