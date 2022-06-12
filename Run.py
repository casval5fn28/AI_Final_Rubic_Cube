'''
   @ Alisha KAreemulla
   @ Thulasi Theja K S
'''

'''
    This file Run.py is the code that needs to be executed if we want to solve the rubic cube by using SARSA algorithm.
    Currently this code runs for the 2*2*2 rubik cube size.
    Once the user runs the Run.py file, it prompts the user to enter the number of times they want to 
    shuffle the Rubik's cube to define the start state.
    Intially the rubik cube is the solved one and after taking input from the user, it gets scrambled.
    One can enter python Run.py in the command prompt to see the output.
'''

import Cube_State as Cubes, MDP, sys, random

def test():

    initial_state = Cubes.GOAL_STATE
    action_ops = Cubes.action_to_operator_dict
    print("The goal state is :\n")
    print(initial_state, "\n")

    shuffles = input("Enter the number of times you want to shuffle the solved Rubik's cube: ")
    for i in shuffles:
        action = random.choice(Cubes.ACTIONS)
        initial_state = action_ops[action].state_transf(initial_state)

    print("The new initial state is :\n")
    print(initial_state, "\n")

    # Initialize MDP and initialize the appropriate class variables.
    rubik_MDP = MDP.MDP()
    rubik_MDP.register_start_state(initial_state)
    rubik_MDP.register_actions(Cubes.ACTIONS)
    rubik_MDP.register_operators(Cubes.OPERATORS)
    rubik_MDP.register_reward_function(Cubes.R)
    rubik_MDP.register_goal_state(Cubes.GOAL_STATE)
    rubik_MDP.register_goal_state_check(Cubes.goalStateCheck)
    rubik_MDP.register_features([Cubes.one_side, Cubes.getCompleteFacesCount])
    rubik_MDP.register_action_to_op(Cubes.action_to_operator_dict)
    rubik_MDP.register_weights([0, 0])

    print("********* SARSA LEARNING *********")
    # Parameters--> discount, episodes, epsilon, learning_rate
    rubik_MDP.sarsaLearning(0.9, 100, 0.2, 0.05)

test()