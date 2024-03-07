
import ast
import json
import networkx as nx


import re
import ast

def parse_xml_to_matched_list(xml_string):
    final_answers = []
    reasonings = []

    # Regular expressions to find content within <final_answer> and <reasoning> tags
    final_answer_pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)
    reasoning_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
    
    # Extract all occurrences of final answers
    final_answer_matches = final_answer_pattern.findall(xml_string)
    # print("final_answer_matches:", final_answer_matches)
    for match in final_answer_matches:
        match = match.strip()
        if '{' in match and '}' in match:
            # Attempt to parse dictionary-like content
            try:
                dic_start = match.index('{')
                dic_end = match.index('}') + 1
                parsed = ast.literal_eval(match[dic_start:dic_end])
                final_answers.append(parsed)
            except (ValueError, SyntaxError):
                # Handle cases where parsing fails
                final_answers.append({'Error': 'Invalid format or incomplete data'})
        else:
            # Directly add non-dictionary-like or placeholder content
            final_answers.append({'Path': match})

    # Extract all occurrences of reasoning
    reasoning_matches = reasoning_pattern.findall(xml_string)
    for match in reasoning_matches:
        reasonings.append(match.strip())

    # Match reasoning with answers into a list of dictionaries
    matched_list = []
    min_length = min(len(final_answers), len(reasonings))
    for i in range(min_length):
        matched_list.append({'reasoning': reasonings[i], 'final_answer': final_answers[i]})

    return matched_list


def parse_xml_to_dict(xml_string):
    try:
        # Ensure both <final_answer> and <reasoning> tags are present
        assert '<final_answer>' in xml_string
        assert '</final_answer>' in xml_string
        assert '<reasoning>' in xml_string 
        assert '</reasoning>' in xml_string
        
        # Extract final answer
        final_answer_start = xml_string.index('<final_answer>') + len('<final_answer>') 
        final_answer_end = xml_string.index('</final_answer>')
        final_answer_element  = xml_string[final_answer_start:final_answer_end].strip()
        
        # Extract reasoning
        reasoning_start = xml_string.index('<reasoning>') + len('<reasoning>')
        reasoning_end = xml_string.index('</reasoning>')
        reasoning_element = xml_string[reasoning_start:reasoning_end].strip()
        
        # Ensure final answer element contains a dictionary format
        assert '{' in final_answer_element
        assert '}' in final_answer_element
        dic_start = final_answer_element.index('{')
        dic_end = final_answer_element.index('}') + 1
        final_answer_element = final_answer_element[dic_start:dic_end].strip()

        try:
            # Attempt to convert final answer string to dictionary
            final_answer_element = ast.literal_eval(final_answer_element)
        except ValueError:
            final_answer_element = ''
            reasoning_element = xml_string  # Fallback to original XML string if conversion fails
    except AssertionError:
        final_answer_element = ''
        reasoning_element = ''

    return final_answer_element, reasoning_element

def ssp_optimal_solution(instance, source, target):
    """Provides the optimal solution for the SSP instance.

    :param instance: The SSP instance as a dictionary with 'nodes' and 'edges'.
    :param source: The source node.
    :param target: The destination node.
    :return: The optimal shortest path length and path.
    """
    G = nx.Graph()
    G.add_nodes_from(instance['nodes'])
    G.add_weighted_edges_from([(edge['from'], edge['to'], edge['weight']) for edge in instance['edges']])
    shortest_path_length = None
    shortest_path = None
    if nx.has_path(G, source=source, target=target):
        shortest_path_length = nx.shortest_path_length(G, source=source, target=target, weight='weight')
        shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
    return shortest_path_length, shortest_path


# SPP
def spp_check(instance, solutions, start_node=None, end_node=None):
    for solution in solutions:
        is_valid, message = spp_check_single(instance, solution, start_node, end_node)
        if is_valid:
            return True, message
    return False, "All solutions are invalid."


def spp_check_single(instance, solution, start_node=None, end_node=None):
    """Validate the solution of the SPP problem.

    :param instance: The instance dictionary with nodes and edges.
    :param solution: The solution dictionary with the path and total distance.
    :param start_node: The start node.
    :param end_node: The end node.
    :return: A tuple of (is_correct, message).
    """

    # Get the start and end nodes
    # Curently, the start and end nodes are the first and last nodes in the instance
    if start_node is None:
        start_node = instance['nodes'][0]
    if end_node is None:
        end_node = instance['nodes'][-1]

    # Convert solution to dictionary
    try:
        path_string = solution.get('Path', '')
        cost_string = solution.get('TotalDistance', '')
    except:
        return False, "The solution is not a dictionary."

    # Calculate the optimal solution
    ssp_optimal_length, ssp_optimal_path = ssp_optimal_solution(instance, start_node, end_node)
    if ssp_optimal_length is None:
        if isinstance(cost_string, int) or cost_string.isdigit():
            return False, f"No path between from node {start_node} to node {end_node}."
        else:
            return True, "No path found from node {start_node} to node {end_node}."

    try:
        path = list(map(int, path_string.split('->')))
        total_cost = int(cost_string)
    except:
        return False, "The solution is not a valid dictionary."

    # Check if path starts and ends with the correct nodes
    if not path or path[0] != start_node or path[-1] != end_node:
        return False, "The path does not start or end at the correct nodes."

    # Check if the path is continuous and calculate the cost
    calculated_cost = 0
    is_in_edge = lambda edge, from_node, to_node: (edge['from'] == from_node and edge['to'] == to_node) or (edge['from'] == to_node and edge['to'] == from_node)
    for i in range(len(path) - 1):
        from_node, to_node = path[i], path[i + 1]
        edge = next((edge for edge in instance['edges'] if is_in_edge(edge, from_node, to_node)), None)

        if not edge:
            return False, f"No edge found from node {from_node} to node {to_node}."

        calculated_cost += edge['weight']

    # Check if the calculated cost matches the total cost provided in the solution
    if calculated_cost != total_cost:
        return False, f"The calculated cost ({calculated_cost}) does not match the provided total cost ({total_cost})."

    if calculated_cost != ssp_optimal_length:
        spp_optimal_path = '->'.join(map(str, ssp_optimal_path))
        return False, f"The calculated cost ({calculated_cost}) does not match the optimal solution ({ssp_optimal_length}): {ssp_optimal_path}."

    return True, "The solution is valid."














# def spp_check(instance, solution, start_node=None, end_node=None):
#     """Validate the solution of the SPP problem.

#     :param instance: The instance dictionary with nodes and edges.
#     :param solution: The solution dictionary with the path and total distance.
#     :param start_node: The start node.
#     :param end_node: The end node.
#     :return: A tuple of (is_correct, message).
#     """
#     # Get the start and end nodes
#     # Curently, the start and end nodes are the first and last nodes in the instance
#     if start_node is None:
#         start_node = instance['nodes'][0]
#     if end_node is None:
#         end_node = instance['nodes'][-1]

#     # Convert solution to dictionary
#     try:
#         path_string = solution.get('Path', '')
#         cost_string = solution.get('TotalDistance', '')
#     except:
#         return False, "The solution is not a dictionary."

#     # Calculate the optimal solution
#     ssp_optimal_length, ssp_optimal_path = ssp_optimal_solution(instance, start_node, end_node)
#     if ssp_optimal_length is None:
#         if isinstance(cost_string, int) or cost_string.isdigit():
#             return False, f"No path between from node {start_node} to node {end_node}."
#         else:
#             return True, "No path found from node {start_node} to node {end_node}."

#     try:
#         path = list(map(int, path_string.split('->')))
#         total_cost = int(cost_string)
#     except:
#         return False, "The solution is not a valid dictionary."

#     # Check if path starts and ends with the correct nodes
#     if not path or path[0] != start_node or path[-1] != end_node:
#         return False, "The path does not start or end at the correct nodes."

#     # Check if the path is continuous and calculate the cost
#     calculated_cost = 0
#     is_in_edge = lambda edge, from_node, to_node: (edge['from'] == from_node and edge['to'] == to_node) or (edge['from'] == to_node and edge['to'] == from_node)
#     for i in range(len(path) - 1):
#         from_node, to_node = path[i], path[i + 1]
#         edge = next((edge for edge in instance['edges'] if is_in_edge(edge, from_node, to_node)), None)

#         if not edge:
#             return False, f"No edge found from node {from_node} to node {to_node}."

#         calculated_cost += edge['weight']

#     # Check if the calculated cost matches the total cost provided in the solution
#     if calculated_cost != total_cost:
#         return False, f"The calculated cost ({calculated_cost}) does not match the provided total cost ({total_cost})."

#     if calculated_cost != ssp_optimal_length:
#         spp_optimal_path = '->'.join(map(str, ssp_optimal_path))
#         return False, f"The calculated cost ({calculated_cost}) does not match the optimal solution ({ssp_optimal_length}): {ssp_optimal_path}."

#     return True, "The solution is valid."

# # Example usage:
# # Define an example SPP instance
# spp_instance = {
#     'nodes': [0, 1, 2, 3],
#     'edges': [
#         {'from': 0, 'to': 1, 'weight': 4},
#         {'from': 1, 'to': 2, 'weight': 1},
#         {'from': 2, 'to': 3, 'weight': 3},
#         {'from': 0, 'to': 3, 'weight': 6}
#     ],
#     'complexity_level': 1
# }

# # Define a solution for the SPP instance
# spp_solution = {
#     'Path': "0->1->2->3",
#     'TotalDistance': 8
# }

# # Validate the solution
# is_valid, message = spp_check(spp_instance, spp_solution, start_node=0, end_node=3)
# print(is_valid, message)