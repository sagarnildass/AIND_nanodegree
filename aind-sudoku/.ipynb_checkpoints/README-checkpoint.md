# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: In the naked twin strategy, we look for two boxes with identical possible values in the same unit. Suppose ['F1':'12', 'I1': '12'], then we can conclude that 1 and 2 must be in F1 and I1 even though we don't know which goes where. So from this assumption, we can eliminate 1 and 3 from all the boxes in the unit corresponding to the twin boxes. So now, if we implement the only choice strategy, it would have much stricter constraints to fill in the possible choices and hence it might lead to a possible solution.

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Diagonal sudoku is a special type of sudoku where among the two main diagonals, the numbers 1 to 9 should all appear exactly once. So here we can implement constraint propagation by adding two additional diagonal unit which accounts for the diagonal constraints. 

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - Fill in the required functions in this file to complete the project.
* `test_solution.py` - You can test your solution by running `python -m unittest`.
* `PySudoku.py` - This is code for visualizing your solution.
* `visualize.py` - This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the `assign_value` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login) for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

