import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('comicsans', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
DARK_GREEN = (1,50,32) # Dark green
LIGHT_GREEN = (0,200,0)

BLOCK_SIZE = 20

class SnakeGameRL:

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.Reset()

    def IsCollision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.width - BLOCK_SIZE or point.x < 0 or point.y > self.height - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False
        
    def Reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.apple = None
        self.PlaceApple()
        self.frame_iteration = 0


    def PlaceApple(self):
        x = random.randint(0, (self.width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.apple = Point(x, y)
        if self.apple in self.snake:
            self.PlaceApple()    
        
    def DrawGrid(self):
        rows = self.height // BLOCK_SIZE
        cols = self.width // BLOCK_SIZE
        sizeBetweenX = self.width // cols
        sizeBetweenY = self.height // rows
        x = 0
        y = 0
        for l in range(rows):
            y += sizeBetweenY
            pygame.draw.line(self.display, LIGHT_GREEN,(0,y), (self.width,y))
            
        for temp in range(cols):
            x += sizeBetweenX
            pygame.draw.line(self.display, LIGHT_GREEN,(x,0), (x,self.width))
            
    def DrawEyes(self):
        centre = BLOCK_SIZE//2
        radius = 3
        left_eye_middle = (self.head.x+centre-radius,self.snake[0].y+8) 
        right_eye_middle = (self.head.x + BLOCK_SIZE -radius*2, self.snake[0].y+8) 
        
        # draws cornea
        pygame.draw.circle(self.display, WHITE, left_eye_middle, radius)
        pygame.draw.circle(self.display, WHITE, right_eye_middle, radius)
        
        # draw pupil
        pygame.draw.circle(self.display, BLACK, left_eye_middle, 1)
        pygame.draw.circle(self.display, BLACK, right_eye_middle, 1)

    def PrintScreen(self):
        self.display.fill(DARK_GREEN)
        
                
        for point in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x+4, point.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
        
        self.DrawEyes()

        pygame.draw.rect(self.display, RED, pygame.Rect(self.apple.x, self.apple.y, BLOCK_SIZE, BLOCK_SIZE))
        self.DrawGrid()
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def Move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx] # right turn (r -> d -> l -> u)
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx] # left turn (r -> u -> l -> d)

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
    def Play(self, action, speed=10, show=True):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self.Move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        is_over = False
        if self.IsCollision() or self.frame_iteration > 100*len(self.snake):
            is_over = True
            reward = -1
            return reward, is_over, self.score

        # 4. place new apple or just move
        if self.head == self.apple:
            self.score += 1
            reward = 1
            self.PlaceApple()
        else:
            self.snake.pop()
        if show:
            # 5. update ui and clock
            self.PrintScreen()
            self.clock.tick(speed)
            
        # 6. return game over and score
        return reward, is_over, self.score