"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # always score terminal states first
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # increase aggression towards end game
    aggression = 1.
    percent_over = float(len(game.get_blank_spaces())) / (game.width * game.height)
    if percent_over <= .5:
        aggression = 1.15
    elif percent_over <= .25:
        aggression = 1.25
    elif percent_over <= .10:
        aggression = 1.5

    return player_moves - (aggression * opponent_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # always score terminal states first
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    # get legal moves for both players
    player_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # check if there is one stealable move
    stealmove_bonus = 0.
    if len(set(player_moves).intersection(opp_moves)) == 1:
      if game.active_player == player:
          stealmove_bonus += 1.
      else:
          stealmove_bonus -= 1.

    return len(player_moves) - len(opp_moves) + stealmove_bonus


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # add reward for central board positions
    position_bonus = 0
    x, y = game.get_player_location(player)
    x_central = float(x) / game.width
    y_central = float(y) / game.height
    if .333 <= x_central or x_central <= .667:
      position_bonus += .5
    if .333 <= y_central or y_central <= .667:
      position_bonus += .5

    # also reward when opponent is non-central
    opp_x, opp_y = game.get_player_location(game.get_opponent(player))
    x_opp_central = float(opp_x) / game.width
    y_opp_central = float(opp_y) / game.height
    if .333 > x_opp_central or x_opp_central > .667:
      position_bonus += .5
    if .333 > y_opp_central or y_opp_central > .667:
      position_bonus += .5

    return player_moves - opponent_moves + position_bonus


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self._minimax(game, depth)[0]

    def _minimax(self, game, depth):
        """Actual implementation of minimax with more convenient return type.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        ((int, int), int)
            Tuple of best move found and its score, although best move found
            will be None for leaf nodes of search and (-1, -1) for nodes where
            no moves are possible (since the game is over)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # base case- if reaching max depth, return score for game state
        if depth == 0:
            return (None, self.score(game, self))

        # keep track of best move found, if no legal moves will be (-1, -1)
        best_move_found = (-1, -1)

        # detect which player is active to determine if at min or max node
        # f is evaluation function for comparing moves (either min or max)
        # v is value of best move found so far
        if game.active_player == self:
            v = float('-inf')
            f = max
        else:
            v = float('inf')
            f = min

        # loop over all legal moves for current player
        for move in game.get_legal_moves():
            # generate new game with that move applied
            forecast = game.forecast_move(move)
            # recursively call minimax at lower depth for forecasted game
            forecast_v = self._minimax(forecast, depth - 1)[1]
            # if forecasted game satisfies evaluation function, record it
            if f(v, forecast_v) == forecast_v:
                best_move_found = move
                v = f(v, forecast_v)

        return (best_move_found, v)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def checktimer(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

            
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # Keep looking deeper until out of time
        depth = 1
        while True:
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, depth)
                depth += 1

            # Break out if running out of time to return best move found so far
            except SearchTimeout:
                break

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Set max value to the smallest possible value
        max_value = float("-inf")
        # returns value if no legal moves exist
        best_move = (-1, -1)
        if len(game.get_legal_moves(self)) >= 1:
            legal_moves = game.get_legal_moves(self)
            best_move = legal_moves[0]
        else:
            return best_move
        # Iterate over all possible candidate moves
        for cand_move in game.get_legal_moves(self):
            # Obtain copy of game.
            cand_game = game.forecast_move(cand_move)
            cand_value = self.min_value(cand_game, depth-1)
            # Update best_move and max_value if cand_value has max value
            if cand_value > max_value:
                best_move, max_value = cand_move, cand_value
        return best_move

    def max_value(self, game, depth):
        """Helper function to minimax that returns the highest value from every
        possible response from the min_value function.  Contigent upon depth.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game 'Board' class representing the
            current candidate game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        int
            The maximum score value for the input candidate game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            # Return Leaf node score
            return self.score(game, self)
        value = float("-inf")
        for move in game.get_legal_moves(game.active_player):
            # Set value equal to the maximum min_value
            value = max(value, self.min_value(game.forecast_move(move), depth-1))
        return value


    def min_value(self, game, depth):
        """Helper function to minimax that returns the lowest value from every
        possible response from the max_value function.  Contigent upon depth.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game 'Board' class representing the
            current candidate game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        int
            The minimum score value for the input candidate game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            # Return leaf node score
            return self.score(game, self)
        value = float("inf")
        for move in game.get_legal_moves(game.active_player):
            # Set value equal to the minimum max_value
            value = min(value, self.max_value(game.forecast_move(move), depth-1))
        return value

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        if len(game.get_legal_moves()) >= 1:
            legal_moves = game.get_legal_moves()
            best_move = legal_moves[0]
        else:
            return best_move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for depth in range(1, len(game.get_blank_spaces())):
            #for depth in range(1, 99999):
                new_move = self.alphabeta(game, depth)
                # Reached end game
                if new_move == ():
                    return best_move
                else:
                    # Update best_move with next deepest iteration
                    best_move = new_move

        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Set max value to the smallest possible value
        best_value = float("-inf")
        # returns empty move if no legal moves exist
        best_move = ()
        # Iterate over all possible candidate moves
        for cand_move in game.get_legal_moves():
            # Obtain copy of game.
            cand_game = game.forecast_move(cand_move)
            cand_value = self.min_value(cand_game, depth-1, alpha, beta)
            # Update best_move and max_value if cand_value has max value
            if cand_value > best_value:
                best_move, best_value = cand_move, cand_value
            # Best move found.
            if best_value >= beta:
                break
            # Update lower bound for pruning
            alpha = max(alpha, best_value)
        return best_move

    def max_value(self, game, depth, alpha, beta):
        """Return max value move of candidate game state.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        int
            The max value move of candidate game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        value = float("-inf")

        for move in game.get_legal_moves():
            value = max(value, self.min_value(game.forecast_move(move), depth-1, alpha, beta))
            # Update lower bound
            alpha = max(alpha, value)
            if value >= beta:
                return value # Found candidate upper value
        return value

    def min_value(self, game, depth, alpha, beta):
        """Return minimum value move of candidate game state.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        int
            The minimum value move of candidate game state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        value = float("inf")

        for move in game.get_legal_moves():
            value = min(value, self.max_value(game.forecast_move(move), depth-1, alpha, beta))
            if value <= alpha:
                return value # Found candidate lower value
            # Update upper bound
            beta = min(beta, value)
        return value