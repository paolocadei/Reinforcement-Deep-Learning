import numpy as np
import matplotlib.pyplot as plt

# GridWorld
class GridWorld(object):
    """
    GridWorld object. Default grid_size (5,5). Default terminal states: fire, water.
    """

    def __init__(self, default=True, grid_size=(6,9), nblocks=10, seed=None):
        
        if default:
            self.grid_size = (6,9)
            self.nstates = np.prod(self.grid_size) # number of states
            self.state_space = np.arange(self.nstates)
            self.blocks = np.array([7,16,25,11,20,29,41])
            self.initial_state = 18
            self.terminal_state = 8
        else:
            self.grid_size = grid_size
            self.nstates = np.prod(self.grid_size) # number of states
            self.state_space = np.arange(self.nstates)
            if seed is not None:
                np.random.seed(seed)
            special_states = np.random.choice(self.state_space, size=(nblocks + 2,))
            self.blocks = special_states[:-2]
            self.initial_state = special_states[-2]
            self.terminal_state = special_states[-1]

        self.nrows, self.ncolumns = self.grid_size

        # reward per step
        self.step_reward = 0.0

        # possible actions
        self.actions = ['U', 'D', 'L', 'R']
        self.action_space = {'U': -self.ncolumns, 'D': self.ncolumns, 'L': -1, 'R': 1}
        self.actions_to_idx = {'U': 0, 'D': 1, 'L': 2, 'R': 3}

    def interact(self, s, a):
        """
        Interaction of agent with the environment.

        Return:
            s_new (int): new state.
            reward (float): reward.
        """

        s_new = s + self.action_space[a]

        reward = 0.0
        if self.transition_check(s, s_new):
            if s_new in self.blocks:
                s_new = s
            elif s_new == self.terminal_state:
                reward = 1.0
        else:
            s_new = s

        return s_new, reward

    def transition_check(self, s, s_new):
        """
        Check if transition from state s to state s_new is valid, i.e., if s_new is not outside
        gridworld.

        Return:
            bool: True if the transition is valid.
        """

        if s_new in self.state_space: # transition to existing state

            if s//self.ncolumns != s_new//self.ncolumns: # transition between rows

                if s%self.ncolumns == s_new%self.ncolumns: # vertical transitions only
                    transition = True
                else:
                    transition = False
            else:
                transition = True
        else:
            transition = False

        return transition
    
# value functions
def initialize_Q(gridworld, random=False, seed=None):
    """
    Randomly generates an action-value function.

    Returns:
        ndarray: 2-dimensional array of size N x 4, where N is the number of states of gridworld. Each
        element is the value of that state-action pair.
    """

    if seed is not None:
        np.random.seed(seed)

    if random:
        Q = np.random.normal(size=(gridworld.nstates, 4), scale=10)
        for s in gridworld.blocks:
            Q[s] = [0.0, 0.0, 0.0, 0.0]
        Q[gridworld.terminal_state] = [0.0, 0.0, 0.0, 0.0]
    else:
        Q = np.zeros((gridworld.nstates, 4))

    return Q

def compute_V(policy, Q):

    return np.sum(policy * Q, axis=1)

# policies random generator
def generate_random_policy(gridworld, deterministic=False, seed=None):
    """
    Randomly generates a deterministic or probabilistic policy pi(a|s).

    Returns:
        ndarray: 2-dimensional array of size (N x 4), where N is the number of states of gridworld.
        The 4 columns correspond, respectively, to the actions "U", "D", "L", "R".
    """

    if seed is not None:
        np.random.seed(seed)
    
    if deterministic:
        probs = np.zeros((gridworld.nstates, 4))
        probs[np.arange(gridworld.nstates), np.random.randint(4, size=(gridworld.nstates,))] = 1.0
    else:
        logits = np.random.normal(size=(gridworld.nstates, 4), scale=1)
        exp = np.exp(logits)
        probs = (exp.T / np.sum(exp, axis=1)).T

    for s in gridworld.blocks:
        probs[s] = [0.0, 0.0, 0.0, 0.0]
    probs[gridworld.terminal_state] = [0.0, 0.0, 0.0, 0.0]

    return(probs)

# auxiliary (print) functions
def color_special_states_func(gridworld, grid_):

    for s in gridworld.blocks:
        idx = np.unravel_index(s, gridworld.grid_size)
        grid_[idx] = 0.5, 0.5, 0.5, 1.0
    
    # initial state
    idx = np.unravel_index(gridworld.initial_state, gridworld.grid_size)
    grid_[idx] = 1.0, 0.5, 0.1, 1.0

    # terminal state
    idx = np.unravel_index(gridworld.terminal_state, gridworld.grid_size)
    grid_[idx] = 0.0, 0.5, 0.8, 1.0
    
    return grid_

def print_grid(gridworld, print_states=False, color_special_states=True, ax=None):

    # Each state is a white, transparent tile
    grid_ = np.ones((*(gridworld.grid_size), 4))
    grid_[:,:,-1] = 0.0

    # Color terminal states
    if color_special_states:
        grid_ = color_special_states_func(gridworld, grid_)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(grid_)

    # Minor ticks
    ax.set_xticks(np.arange(-.5, gridworld.ncolumns, 1), minor=True)
    ax.set_yticks(np.arange(-.5, gridworld.nrows, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    # Remove other ticks and labels
    ax.tick_params(which='major', bottom=False, left=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Print states
    if print_states:
        for j in range(gridworld.nstates):
            if j==gridworld.initial_state:
                ax.text(j%gridworld.ncolumns, j//gridworld.ncolumns, 'S', ha="center", va="center")
            elif j==gridworld.terminal_state:
                ax.text(j%gridworld.ncolumns, j//gridworld.ncolumns, 'G', ha="center", va="center")
            else:
                ax.text(j%gridworld.ncolumns, j//gridworld.ncolumns, j, ha="center", va="center")

def print_v(v, gridworld, ax=None, color_special_states=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
        print_grid(gridworld, ax=ax, color_special_states=color_special_states, **kwargs)

    cmap = plt.cm.get_cmap('Greens')
    norm = plt.Normalize(v.min(), v.max())
    grid_ = cmap(norm(v)).reshape(*(gridworld.grid_size), 4)
    
    # Color terminal states
    if color_special_states:
        grid_ = color_special_states_func(gridworld, grid_)

    ax.imshow(grid_)

    # Print state values
    for j in range(gridworld.nstates):
        if j not in gridworld.blocks:
            ax.text(j%gridworld.ncolumns, j//gridworld.ncolumns, np.round(v[j],2), ha="center", va="center", size=6)

def print_policy(policy, gridworld, ax=None, color_special_states=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
        print_grid(gridworld, ax=ax, color_special_states=color_special_states, **kwargs)

    # Print policy
    for j in range(gridworld.nstates):
        if j not in gridworld.blocks and j!=gridworld.terminal_state:
            for direction_idx in range(4):
                prob = policy[j, direction_idx]
                direction = [(0.0, -0.25), (0.0, 0.25), (-0.25, 0), (0.25, 0)][direction_idx] # U, D, L R
                ax.arrow(j%gridworld.ncolumns, j//gridworld.ncolumns, *direction, head_width=0.1, alpha=prob)

def print_overview(gridworld, v, policy):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    ax = axes[0]
    print_grid(gridworld, ax=ax, print_states=True)
    ax.set_title("States")

    ax = axes[1]
    print_grid(gridworld, ax=ax)
    print_v(v, gridworld, ax=ax)
    ax.set_title("State values")

    ax = axes[2]
    print_grid(gridworld, ax=ax)
    print_policy(policy, gridworld, ax=ax)
    ax.set_title("Policy");