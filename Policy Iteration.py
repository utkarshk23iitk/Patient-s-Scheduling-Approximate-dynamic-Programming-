import numpy as np

# Possible states
width, height = 3, 4
states = list(range(width * height))
terminal_states = [11]
actions = ['U', 'D', 'L', 'R']
discount = 0.8
theta = 1e-3

# 0 1 2
# 3 4 5

def transition_function(state, action):
    if state in terminal_states:
        return [state, 0]
    
    x, y = state%width, state//width

    if action == "U":
        y = max(y-1, 0)
        reward = -1

    elif action == "D":
        y = min(y + 1, height - 1)
        reward = -1

    elif action == "L":
        x = max(0, x - 1)
        reward = -1
        
    elif action == "R":
        x = min(width - 1, x + 1)
        reward = -1

    new_state = y*width + x
    return [new_state, reward]

## Finding the value function for a random initial policy
policy = ['R' if s not in terminal_states else '-' for s in states]
value = np.zeros(len(states))
stable_policy = False

while (not(stable_policy)):
    while True:
        delta = 0
        new_value = value.copy()
        for s in states:
            if s in terminal_states:
                continue
            else:
                a = policy[s]
                t = transition_function(s,a)
                new_v = t[1] + discount*(value[t[0]])
                delta = max(delta, abs(new_v - value[s]))
                new_value[s] = new_v
        value = new_value
        if (delta < theta):
            break

    ## Policy improvement
    stable_policy = True
    for s in states:
        if s in terminal_states:
            continue
        else:
            old_action = policy[s]
            best_action = None
            best_value = float('-inf')
            for a in actions:
                t = transition_function(s,a)
                q = t[1] + discount*(value[t[0]])
                if q > best_value:
                    best_value = q
                    best_action = a
            policy[s] = best_action
            if old_action != best_action:
                stable_policy = False


value = value.reshape(height, width)
val_mat = np.asmatrix(value)
print(f'Optimal Value function: \n{val_mat}')

policy = np.array(policy)
policy = policy.reshape(height, width)
pol_mat = np.asmatrix(policy)
print(f'Optimal Policy:\n {pol_mat}')
