'''
Markov Decision Process is used for solving the Rubik's cube. 
SARSA is implemented to enable state space exploration. 
SARSA's agent interacts with its environment and updates policy for based on the action taken.
Feature engineering is performed to select features to update Q values using linear function approximation.

Epsilon greedy algorithm is used to decide the first action and once the agent is in the new state, 
the choice of action is based on the best policy action for that state.

For Q-Learning , all state action combinations are known and updates to the policy happen on these values.
'''

import random
import itertools

REPORTING = True

class MDP:
    def __init__(self):
        self.known_states = set()
        self.q = {}

    # registering the start state
    def register_start_state(self, start_state):
        self.start_state = start_state
        self.current_state = start_state

    # Register actions
    def register_actions(self, action_list):
        self.actions = action_list

    # Register reward function        
    def register_reward_function(self, reward):
        self.R = reward

    # Register operators
    def register_operators(self, op_list):
        self.ops = op_list
        
    # Register goal state
    def register_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Check if we reach the goal state 
    def register_goal_state_check(self, goal_state_check):
        self.goal_state_check = goal_state_check
        
    # List of features
    def register_features(self, features):
        self.features = features
        
    # Registers the weights for features
    def register_weights(self, weights):
        self.weights = weights

    # Defines what action to do
    def register_action_to_op(self, action_to_op):
        self.action_to_op = action_to_op

    # Here the parameters are discount->0.9, n_episodes-> 100, epsilon->0.2, learning_rate->0.05
    def sarsaLearning(self, discount, n_episodes, epsilon, learning_rate):
        # tmp is assumed to be 1
        tmp = 1
        # Counter to track how many episodes got to the goal state
        episode_goal_count = 0
        for j in range(n_episodes):
            s = self.current_state
            best_action = None
            # choose an action 'a' based on epsilon greedy policy
            random_num = random.random()
            if random_num > epsilon:
                m_val = -1000
                for a in self.actions:
                    if (s, a) in self.q:
                        if self.q[(s, a)] > m_val:
                            m_val = self.q[(s, a)]
                            best_action = a
                if best_action == None:
                    best_action = random.choice(self.actions)
            else:
                best_action = random.choice(self.actions)
            count = 0

            # The stopping condition is to check for the goal state
            while not s == self.goal_state:
                count += 1
                # Use a found earlier to move to 's_prime'
                s_prime = self.action_to_op[best_action].state_transf(s)
                # Get the reward for the (s, a, 's_prime') transition
                reward = self.R(s, best_action, s_prime)
                # Policy based best action for 'a_prime'
                best_action_prime = None
                m_val = -1000
                for a_prime in self.actions:
                    if (s_prime, a_prime) in self.q:
                        if self.q[(s_prime, a_prime)] > m_val:
                            m_val = self.q[(s_prime, a_prime)]
                            best_action_prime = a_prime

                if best_action_prime == None:
                    best_action_prime = random.choice(self.actions)
                # Q-values are updated based on Linear function approximation
                self.q[(s_prime, best_action_prime)] = tmp + self.weights[0]*self.features[0](s_prime) + self.weights[1]*self.features[1](s_prime) 
                self.q[(s, best_action)] = tmp + self.weights[0]*self.features[0](s) + self.weights[1]*self.features[1](s)
                # Calculate delta
                delta = reward + discount*self.q[(s_prime, best_action_prime)] - self.q[(s, best_action)]
                weight_sum = 0
                
                # Update weights
                for i in range(len(self.features)):
                    self.weights[i] += learning_rate*delta*self.features[i](s)
                    weight_sum += self.weights[i]
                tmp = learning_rate*delta*1 # f0 is 1 by default
                weight_sum = tmp + sum(self.weights)
                # Normalize weights as the sum of weights for all features can't be more than 1.
                tmp = tmp/weight_sum
                for i in range(len(self.features)):
                    self.weights[i] = self.weights[i]/weight_sum
                    
                # Make a = a_prime ,s = s_prime
                s = s_prime
                best_action = best_action_prime
                if self.goal_state_check(s): # Alternate orientations of the goal state are also goal states
                    print("====================================")
                    print("\nGoal state reached for episode ", j)
                    print("Action taken to reach s_prime ", best_action)
                    print("Q value for (s, a) ", self.q[(s, best_action)])
                    print("Goal state reached ", s, "\n")
                    episode_goal_count+=1
                    break

                ''' Restrict number of moves to 50 per episode , since some orientations 
                    of the start state may take a long time getting to the goal state'''
                if count > 50:
                    break

        print("The goal state was reached in ", episode_goal_count, "episodes.")
