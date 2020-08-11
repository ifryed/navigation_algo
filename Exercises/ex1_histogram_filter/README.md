# Ex1
## Objective
Write a function that performs Histogram filter in two dimensions

## Solution
The solution is in `main.py` and `histogram.py'. 
To execute:

    > python main.py
    
## Bonus
The files 'main_game.py' and 'histogram_game.py' contain an extension to the exercise.
This is a programme that simulates a robot trying to reach it's goal.

In 'main_game.py'the user can define the size of the world, which will be 'painted' at random. 
Then the initial starting point and and end point will be drawn.
A plot will appear with the real location of the robot (in green), the location of the goal (in yellow) 
and the estimated robot location (in red). The 'game' ends, when the estimated location reaches the goal.
Each round, the robot advances one step towards the goal. The robot can move in 8-directions.

To execute:

    > python main_game.py