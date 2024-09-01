# Tabu Search for Clever Traveling Salesperson Problem

This script outlines the implementation of a Tabu Search algorithm tailored for solving the Traveling Salesman Problem (TSP) with precedence constraints. This project is my submission for the course 42137 Optimization using Metaheuristics, taught by Thomas Jacob Riis Stidsen and Dario Pacino at DTU.

  **Neighborhood Operator: ** The algorithm uses a custom swap operation to generate neighboring solutions.
                          Specifically, it considers swapping two customers in the current route to explore potential new solutions.
                          The feasibility of these swaps is evaluated by checking whether they violate any precedence constraints.

  **Termination Criteria:**   The search terminates based on a time limit specified by the user.
                          Additionally, it monitors the number of iterations since the last improvement.
                          If this number exceeds a threshold, the algorithm switches between intensification (focusing on improving the current best solution)
                          and diversification (exploring new regions of the solution space).

  **Tabu List Management: **  The algorithm maintains a tabu list to store recently visited moves (i.e., customer swaps) 
                          to prevent cycling and encourage exploration of different parts of the solution space. The tabu list is periodically 
                          cleared and updated to manage the search process effectively.

  **Cost Function: **         The cost of a given route is calculated using a distance matrix (cost matrix).
                          This function is crucial for evaluating and comparing different solutions as the algorithm searches for the optimal route.
                          The objective function minimizes the total travel cost of the route.

  **Initialization: **        The algorithm begins by generating an initial feasible solution using a nearest-neighbor heuristic.
                          This heuristic constructs a route by iteratively selecting the closest unvisited customer, respecting precedence constraints.
                          The initial solution serves as a starting point for the Tabu Search.

  **Validity Check:  **       Before accepting a swap (move), the algorithm checks if it maintains the feasibility of the route.
                          This includes ensuring that the precedence constraints (which may dictate the order in which customers must be visited) are respected.
                          The validity check helps ensure that the generated solutions are not only cost-effective but also adhere to the problem's constraints.

