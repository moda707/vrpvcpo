# Deep RL to solve Vehicle Routing Problem with Stochastic Demands and Partial Outsourcing option

# Problem:
A set of customers register their request to serve their demand (stochastic) every morning. Due to the limited working hours per day, the operator must choose a subset of customers to outsource in the morning and route a fleet of vehicles with a limited capacity to serve the remaining customers in the shortest time. Working out of the regular time costs a penalty per unit of time for the company.

# Method:
Two steps decision: 
1- A simulated annealing algorithm chooses the subset of customers to be outsourced.
2- A deep reinforcement learning method is adopted to route vehicles.

This project uses Graph Neural Network and Transformers to encode the state of customers and vehicles, and uses Q-learning algorithm to develop a single policy for each vehicle to use autonomously.
