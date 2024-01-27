import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameRL, Direction, Point
from model import DQN, QTrainer

import matplotlib.pyplot as plt
from IPython import display

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001 

BLOCK_SIZE = 20

TRAINING_N_GAMES = 200

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.decay = 0.01
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = DQN(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def GetAction(self, state, Train=True):     
        
        # random moves: tradeoff exploration / exploitation        
        final_move = [0,0,0]
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon and Train:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1      
        return final_move


    def GetState(self, game):
        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)
        
        is_direction_left = game.direction == Direction.LEFT
        is_direction_right = game.direction == Direction.RIGHT
        is_direction_up = game.direction == Direction.UP
        is_direction_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (is_direction_right and game.IsCollision(point_right)) or 
            (is_direction_left and game.IsCollision(point_left)) or 
            (is_direction_up and game.IsCollision(point_up)) or 
            (is_direction_down and game.IsCollision(point_down)),

            # Danger right
            (is_direction_up and game.IsCollision(point_right)) or 
            (is_direction_down and game.IsCollision(point_left)) or 
            (is_direction_left and game.IsCollision(point_up)) or 
            (is_direction_right and game.IsCollision(point_down)),

            # Danger left
            (is_direction_down and game.IsCollision(point_right)) or 
            (is_direction_up and game.IsCollision(point_left)) or 
            (is_direction_right and game.IsCollision(point_up)) or 
            (is_direction_left and game.IsCollision(point_down)),
            
            # Move direction
            is_direction_left,
            is_direction_right,
            is_direction_up,
            is_direction_down,
            
            # Apple location 
            game.apple.x < game.head.x,  # apple left
            game.apple.x > game.head.x,  # apple right
            game.apple.y < game.head.y,  # apple up
            game.apple.y > game.head.y  # apple down
            ]

        return np.array(state, dtype=int)

    def Remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def TrainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.TrainStep(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.TrainStep(state, action, reward, next_state, done)

    def TrainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.TrainStep(state, action, reward, next_state, done)    

def Train(agent, game):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    while agent.n_games < TRAINING_N_GAMES:
        # get state
        state = agent.GetState(game)

        # get move
        final_move = agent.GetAction(state)

        # perform move and get next state
        reward, done, score = game.Play(final_move, 100000000, show=False)
        next_state = agent.GetState(game)

        # Train short memory
        agent.TrainShortMemory(state, final_move, reward, next_state, done)

        # Remember
        agent.Remember(state, final_move, reward, next_state, done)

        if done:
            # Train long memory, Plot result
            
            game.Reset()
            agent.n_games += 1
            agent.TrainLongMemory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, "Epsilon: ", str(agent.epsilon))
            
            # ploting data
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            Plot(plot_scores, plot_mean_scores, "Training...")
    return record

def Test(agent, game, record):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    while True:
        # get old state
        state = agent.GetState(game)
        
        state

        # get move
        final_move = agent.GetAction(state, Train=False)

        # perform move and get new state
        reward, done, score = game.Play(final_move, 10)
        next_state = agent.GetState(game)

        if done:
            # Plot result
            game.Reset()
            agent.n_games += 1
            
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            # ploting data
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / (agent.n_games-TRAINING_N_GAMES)
            plot_mean_scores.append(mean_score)
            Plot(plot_scores, plot_mean_scores, "Testing")
          
def Plot(scores, mean_scores, title,):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.Plot(scores)
    plt.Plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    

if __name__ == '__main__':
    agent = Agent()
    game = SnakeGameRL()
    train_record = Train(agent, game)
    Test(agent, game, train_record)
    
