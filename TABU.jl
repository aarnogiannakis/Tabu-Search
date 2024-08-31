# <42137 Optimization using Metaheuristics -- Assignment 04>
# Tabu Search for Clever Traveling Salesperson Problem
#
# This script outlines the implementation of a Tabu Search algorithm tailored for solving the Traveling Salesman Problem (TSP) with precedence constraints. 
#
#   Neighborhood Operator:  The algorithm uses a custom swap operation to generate neighboring solutions.
#                           Specifically, it considers swapping two customers in the current route to explore potential new solutions.
#                           The feasibility of these swaps is evaluated by checking whether they violate any precedence constraints.
#
#   Termination Criteria:   The search terminates based on a time limit specified by the user.
#                           Additionally, it monitors the number of iterations since the last improvement.
#                           If this number exceeds a threshold, the algorithm switches between intensification (focusing on improving the current best solution)
#                           and diversification (exploring new regions of the solution space).
#
#   Tabu List Management:   The algorithm maintains a tabu list to store recently visited moves (i.e., customer swaps) 
#                           to prevent cycling and encourage exploration of different parts of the solution space. The tabu list is periodically 
#                           cleared and updated to manage the search process effectively.
#
#   Cost Function:          The cost of a given route is calculated using a distance matrix (cost matrix).
#                           This function is crucial for evaluating and comparing different solutions as the algorithm searches for the optimal route.
#                           The objective function minimizes the total travel cost of the route.
#
#   Initialization:         The algorithm begins by generating an initial feasible solution using a nearest-neighbor heuristic.
#                           This heuristic constructs a route by iteratively selecting the closest unvisited customer, respecting precedence constraints.
#                           The initial solution serves as a starting point for the Tabu Search.
#
#   Validity Check:         Before accepting a swap (move), the algorithm checks if it maintains the feasibility of the route.
#                           This includes ensuring that the precedence constraints (which may dictate the order in which customers must be visited) are respected.
#                           The validity check helps ensure that the generated solutions are not only cost-effective but also adhere to the problem's constraints.

#*****************************************************************************************************


#*****************************************************************************************************
using Random
Random.seed!(810)

function read_instance(filename)
    # Opens the file
    f = open(filename)
    # Reads the name of the instance
    name = split(readline(f))[2]
    # Reads the upper bound value
    upper_bound = parse(Int64, split(readline(f))[2])
    readline(f) # Type
    readline(f) # Comment
    # Reads the dimensions of the problem
    dimension = parse(Int64, split(readline(f))[2])
    readline(f) # Edge1
    readline(f) # Edge2
    readline(f) # Edge3
    readline(f) # Dimension 2

    # Initializes the cost matrix
    cost_matrix = zeros(Int64, dimension, dimension)
    # Reads the cost matrix
    for i in 1:dimension
        data = parse.(Int64, split(readline(f)))
        cost_matrix[i, :] = data
    end
    # Closes the file
    close(f)

    # Returns the input data
    return dimension, cost_matrix
end
#*****************************************************************************************************


#*****************************************************************************************************
# Immutable struct for the input data
struct TSPInputData
    cost_matrix::Array{Int64, 2}
    dimension::Int64
    start_time::UInt64
    time_limit::Int64
    TSPInputData(cost_matrix, dimension, time_limit) = new(cost_matrix, dimension, time_ns(), time_limit)
end

# A mutable struct for the solution
mutable struct TSPSolution
    route::Array{Int64, 1}
    objective_value::Float64
    TSPSolution(dimension) = new(zeros(Int64, dimension), 0)
end
#*****************************************************************************************************


#*****************************************************************************************************
# This function checks if a customer can be placed in a position in the route
# by checking the precedence with all other customers who are not visited yet
function is_feasible(next_customer, input_data::TSPInputData, visited_customers)
    for i in 1:input_data.dimension
        if !visited_customers[i] && input_data.cost_matrix[next_customer, i] == -1
            return false
        end
    end
    return true
end

# This function is a part of the construction heuristic which finds the
# best customer (least cost to visit) from the list of unvisited customers
function find_next_customer(input_data::TSPInputData, current_customer, visited_customers)
    min_cost = Inf    
    selected_customer = 0
    for i in 1:input_data.dimension
        if !visited_customers[i] && input_data.cost_matrix[current_customer, i] < min_cost && is_feasible(i, input_data, visited_customers)
            selected_customer = i
            min_cost = input_data.cost_matrix[current_customer, i]
        end
    end    
    return selected_customer
end

# This function is the main construction heuristic that builds 
# a feasible route following the nearest neighbor algorithm
function construct_initial_route(start_customer, input_data::TSPInputData)

    route = zeros(Int64, input_data.dimension)
    visited_customers = zeros(Bool, input_data.dimension)
    route[1] = start_customer
    visited_customers[start_customer] = true    
    for i in 1:input_data.dimension - 1
        next_customer = find_next_customer(input_data, route[i], visited_customers)
        route[i + 1] = next_customer
        visited_customers[next_customer] = true
    end
    return route
end

# This function identifies the customer that should be visited first based on
# the number of precedence constraints. It ensures that the first customer is
# selected appropriately, rather than assuming the first one in the instance is correct.
function generate_initial_solution(input_data::TSPInputData)
    initial_solution = TSPSolution(input_data.dimension)
    precedence_count = zeros(Int, input_data.dimension)
    for j in 1:input_data.dimension
        for i in 1:input_data.dimension
            if input_data.cost_matrix[i, j] == -1
                precedence_count[j] += 1
            end
        end
    end

    first_customer = argmax(precedence_count)
    initial_solution.route = construct_initial_route(first_customer, input_data)
    initial_solution.objective_value = compute_cost(initial_solution.route, input_data)
    return initial_solution
end
#*****************************************************************************************************


#*****************************************************************************************************
# This function computes the total cost of the given route
function calculate_route_cost(route, input_data::TSPInputData)
    total_cost = 0
    for i in 1:input_data.dimension - 1
        total_cost += input_data.cost_matrix[route[i], route[i + 1]]
    end
    return total_cost
end

# This function calculates the elapsed time since the start of the algorithm
function compute_elapsed_time(input_data::TSPInputData)
    return round((time_ns() - input_data.start_time) / 1e9, digits=3)
end

# This function checks if the route is feasible by verifying all precedence constraints
function check_route_feasibility(route, input_data::TSPInputData)
    for i in 1:input_data.dimension - 1
        for j in i + 1:input_data.dimension
            if input_data.cost_matrix[route[i], route[j]] == -1
                return false
            end
        end
    end

    return true
end

# This function checks if two customers i and j can be swapped in the given route
function is_swap_feasible(input_data::TSPInputData, route, i, j)
    modified_route = deepcopy(route)
    modified_route[i], modified_route[j] = modified_route[j], modified_route[i]
    
    for k in 1:input_data.dimension - 1
        for l in k + 1:input_data.dimension
            if input_data.cost_matrix[modified_route[k], modified_route[l]] == -1
                return false
            end
        end
    end
    return true
end
#*****************************************************************************************************


#*****************************************************************************************************
# Function to evaluate the change in objective value after swapping two customers
function evaluate_swap_delta(input_data::TSPInputData, solution::TSPSolution, i, j)

    if i == 1 && j == input_data.dimension # Case 1: i is at the beginning and j is at the end
        return solution.objective_value - input_data.cost_matrix[solution.route[i], solution.route[i + 1]] -
               input_data.cost_matrix[solution.route[j - 1], solution.route[j]] +
               input_data.cost_matrix[solution.route[j], solution.route[i + 1]] +
               input_data.cost_matrix[solution.route[j - 1], solution.route[i]]
    elseif i == 1 && j - i == 1 # Case 2: i is at the beginning and j is next to i
        return solution.objective_value - input_data.cost_matrix[solution.route[i], solution.route[j]] -
               input_data.cost_matrix[solution.route[j], solution.route[j + 1]] +
               input_data.cost_matrix[solution.route[j], solution.route[i]] +
               input_data.cost_matrix[solution.route[i], solution.route[j + 1]]
    elseif i == 1 && j - i > 1 # Case 3: i is at the beginning and j is somewhere in the middle
        return solution.objective_value - input_data.cost_matrix[solution.route[i], solution.route[i + 1]] -
               input_data.cost_matrix[solution.route[j - 1], solution.route[j]] -
               input_data.cost_matrix[solution.route[j], solution.route[j + 1]] +
               input_data.cost_matrix[solution.route[j], solution.route[i + 1]] +
               input_data.cost_matrix[solution.route[j - 1], solution.route[i]] +
               input_data.cost_matrix[solution.route[i], solution.route[j + 1]]
    elseif j == input_data.dimension && j - 1 == i # Case 4: j is at the end and i is right before j
        return solution.objective_value - input_data.cost_matrix[solution.route[i - 1], solution.route[i]] -
               input_data.cost_matrix[solution.route[i], solution.route[j]] +
               input_data.cost_matrix[solution.route[i - 1], solution.route[j]] +
               input_data.cost_matrix[solution.route[j], solution.route[i]]
    elseif j == input_data.dimension && j - i > 1 # Case 5: j is at the end and i is in the middle
        return solution.objective_value - input_data.cost_matrix[solution.route[i - 1], solution.route[i]] -
               input_data.cost_matrix[solution.route[i], solution.route[i + 1]] -
               input_data.cost_matrix[solution.route[j - 1], solution.route[j]] +
               input_data.cost_matrix[solution.route[i - 1], solution.route[j]] +
               input_data.cost_matrix[solution.route[j], solution.route[i + 1]] +
               input_data.cost_matrix[solution.route[j - 1], solution.route[i]]
    elseif j - i == 1 # Case 6: Both i and j are in the middle and next to each other
        return solution.objective_value - input_data.cost_matrix[solution.route[i - 1], solution.route[i]] -
               input_data.cost_matrix[solution.route[i], solution.route[j]] -
               input_data.cost_matrix[solution.route[j], solution.route[j + 1]] +
               input_data.cost_matrix[solution.route[i - 1], solution.route[j]] +
               input_data.cost_matrix[solution.route[j], solution.route[i]] +
               input_data.cost_matrix[solution.route[i], solution.route[j + 1]]
    else # Case 7: Both i and j are in the middle
        return solution.objective_value - input_data.cost_matrix[solution.route[i - 1], solution.route[i]] -
               input_data.cost_matrix[solution.route[i], solution.route[i + 1]] -
               input_data.cost_matrix[solution.route[j - 1], solution.route[j]] -
               input_data.cost_matrix[solution.route[j], solution.route[j + 1]] +
               input_data.cost_matrix[solution.route[i - 1], solution.route[j]] +
               input_data.cost_matrix[solution.route[j], solution.route[i + 1]] +
               input_data.cost_matrix[solution.route[j - 1], solution.route[i]] +
               input_data.cost_matrix[solution.route[i], solution.route[j + 1]]
    end
end

# This function returns the best non-tabu neighbor and the corresponding objective value
function find_best_swap(input_data::TSPInputData, solution::TSPSolution, tabu_list)
    best_i = 0
    best_j = 0
    best_obj_value = Inf

    for i in 1:input_data.dimension - 1
        if !(i in tabu_list)
            for j in i + 1:input_data.dimension
                if !(j in tabu_list)
                    if is_swap_feasible(input_data, solution.route, i, j)
                        new_obj_value = evaluate_swap_delta(input_data, solution, i, j)
                        if new_obj_value < best_obj_value
                            best_i = i
                            best_j = j            
                            best_obj_value = new_obj_value
                        end
                    end
                end
            end
        end
    end
    return best_i, best_j, best_obj_value    
end

# This function performs the swap operation on the route
function perform_swap(route, i, j)
    route[i], route[j] = route[j], route[i]
    return route
end
#*****************************************************************************************************


#*****************************************************************************************************
# This function generates a random route from scratch (Diversification step)
function diversify_route(input_data::TSPInputData, solution::TSPSolution)

    # Time checker to ensure there's sufficient time left to perform the diversification
    if compute_elapsed_time(input_data) < input_data.time_limit - 5

        visited_customers = zeros(Bool, input_data.dimension)
        new_route = [solution.route[1]]
        visited_customers[solution.route[1]] = true
        splice!(solution.route, 1)

        while length(solution.route) != 1
            random_customer = rand(solution.route)
            position = findfirst(x -> x == random_customer, solution.route)
            if is_feasible(random_customer, input_data, visited_customers)
                push!(new_route, random_customer)
                splice!(solution.route, position)
                visited_customers[random_customer] = true            
            end
        end

        # Add the last remaining customer to the route
        push!(new_route, solution.route[1])
        solution.route = deepcopy(new_route)
        solution.objective_value = calculate_route_cost(solution.route, input_data)
    end
end

# This function initializes the Tabu list and counter
function initialize_tabu_list_and_counter(K)
    tabu_list = zeros(Int64, K)
    return (tabu_list, 1)
end

# This function updates the Tabu list with new moves
function update_tabu_list(tabu_list, counter, tabu_move)
    tabu_list[counter] = tabu_move
    counter = (counter < length(tabu_list) ? counter + 1 : 1)

    return (tabu_list, counter)
end

# This function clears the Tabu list
function clear_tabu_list(tabu_list)
    for i in 1:length(tabu_list)
        tabu_list[i] = 0
    end
    return (tabu_list, 1)
end

# This is the main Tabu Search function
function perform_tabu_search(input_data::TSPInputData, solution::TSPSolution, tabu_list_length)
    (tabu_list, counter) = initialize_tabu_list_and_counter(tabu_list_length)
    iteration = 0
    best_iteration = 0
    best_solution = deepcopy(solution)
    last_best_solution = deepcopy(solution) # Used to manage the switch between Intensification and Diversification
    
    while compute_elapsed_time(input_data) < input_data.time_limit

        iteration += 1

        swap_i, swap_j, swap_objective = find_best_swap(input_data, solution, tabu_list)

        (tabu_list, counter) = update_tabu_list(tabu_list, counter, swap_i)
        (tabu_list, counter) = update_tabu_list(tabu_list, counter, swap_j)

        perform_swap(solution.route, swap_i, swap_j)
        solution.objective_value = swap_objective
        
        if swap_objective < best_solution.objective_value
            best_solution = deepcopy(solution)
            best_iteration = iteration
        end

        if iteration - best_iteration > input_data.dimension * 10
            
            # If the same best solution is hit after intensification, diversify
            if last_best_solution.route != best_solution.route

                best_iteration = iteration
                clear_tabu_list(tabu_list)
                solution = deepcopy(best_solution)
                last_best_solution = deepcopy(best_solution)
                
            else

                best_iteration = iteration
                clear_tabu_list(tabu_list)
                diversify_route(input_data, solution)
                last_best_solution = deepcopy(best_solution)
                
            end
        end
    end

    return best_solution    
end
#*****************************************************************************************************


#*****************************************************************************************************
# This function generates the output file with the final solution
function write_solution_to_file(filename, route)
    open(filename, "w") do file                
        write(file, join(route, " "))
    end    
end
#*****************************************************************************************************


#*****************************************************************************************************
# Main function to set up and execute the Tabu Search
function main(instance_filename::String, solution_filename::String, time_limit::Int64)
    dimension, cost_matrix = read_instance(instance_filename)
    input_data = TSPInputData(cost_matrix, dimension, time_limit)
    initial_solution = generate_initial_solution(input_data)
    tabu_list_length = convert(Int, round(0.3 * dimension))
    best_solution = perform_tabu_search(input_data, initial_solution, tabu_list_length)
    write_solution_to_file(solution_filename, best_solution.route .-= 1)

end

main(ARGS[1], ARGS[2], parse(Int64, ARGS[3]))