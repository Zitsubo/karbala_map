import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
# made py ZITSUBO 2024/11/13 
# my search algorithms for whatever reason
def BFS(graph, start, goal):
    visited = []
    queue = [[start]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in visited:
            visited.append(node)
        else:
            continue
        if node == goal:
            #print(len(path))
            return path
        adjacent_nodes = graph[node]
        print(adjacent_nodes)
        for node2 in adjacent_nodes:
            new_path = path.copy()
            new_path.append(node2)
            queue.append(new_path)
def DFS(graph, start, goal):
    stack = [[start]]
    visited = []
    while stack:
        path = stack.pop()
        node = path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        adjacent_nodes = graph[node]
        for node2 in adjacent_nodes:
            new_path = path + [node2]
            stack.append(new_path)
def path_cost(path):
    total_cost = 0
    for (node, cost) in path:
        total_cost += cost
    return total_cost, path[-1][0]


def path_distance_calc(graph, path):
    total_distance = 0
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i + 1])
        
        if edge_data:
            for _, data in edge_data.items():
                total_distance += data.get('length', 0) 
        else:
            print(f"No direct edge between {path[i]} and {path[i + 1]}")
    
    return total_distance



import heapq

def UCS(graph, start, goal):
    # Priority queue for UCS (min-heap)
    queue = []
    # Push the start node into the queue with 0 cost
    heapq.heappush(queue, (0, start, [start]))  # (cost, node, path)
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)  # Pop the node with the least cost

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            return path  # Return the path to the goal as a list of node IDs

        # Explore neighbors
        for neighbor, data in graph[node].items():
            # Check if the edge exists
            if neighbor in graph[node]:
                edge_cost = data.get('weight', 1)  # Default weight is 1 if no weight is provided
                if neighbor not in visited:
                    new_cost = cost + edge_cost
                    new_path = path + [neighbor]  # Append only the node ID
                    heapq.heappush(queue, (new_cost, neighbor, new_path))  # Push the new path with updated cost

    return None  # Return None if no path is found


    return None  # Return None if no path is found
                
def DLS(graph, start, goal, limit):
    def recursive_dls(node, goal, path, depth):
        if node in path:
            return
        path.append(node)
        if node == goal:
            return path
        if depth >= limit:
            return None

        for neighbor in graph[node]:
            if neighbor not in path:  # Avoid cycles by checking if neighbor is already in path
                result = recursive_dls(neighbor, goal, path, depth + 1)
                if result is not None:
                    return result  # Goal found in this path
          # Backtrack if goal is not found at this depth
        path.pop()
        return None
    return recursive_dls(start, goal, [], 0)

def IDDFS(graph, start, goal, max_depth):
    for depth in range(max_depth + 1):
        path = DLS(graph, start, goal, depth)
        if path is not None:
            return path
    return None

def BDS(graph, start, goal, method="BFS"):
    forward_frontier = [[start]]
    backward_frontier = [[goal]]
    forward_visited = {start: [start]}
    backward_visited = {goal: [goal]}
    
    while forward_frontier and backward_frontier:
        # Expand forward
        if method == "BFS":
            forward_path = forward_frontier.pop(0)  # FIFO for BFS
        else:
            forward_path = forward_frontier.pop()  # LIFO for DFS
            
        forward_node = forward_path[-1]
        
        for neighbor in graph[forward_node]:
            if neighbor not in forward_visited:
                new_path = forward_path + [neighbor]  # Extend the path
                forward_frontier.append(new_path)
                forward_visited[neighbor] = new_path  # Track full path
                
                if neighbor in backward_visited:
                    # Combine paths, ensuring the common node is not duplicated
                    return forward_visited[neighbor] + backward_visited[neighbor][::-1][1:]

        # Expand backward
        if method == "BFS":
            backward_path = backward_frontier.pop(0)  # FIFO for BFS
        else:
            backward_path = backward_frontier.pop()  # LIFO for DFS
            
        backward_node = backward_path[-1]
        
        for neighbor in graph[backward_node]:
            if neighbor not in backward_visited:
                new_path = backward_path + [neighbor]  # Extend the path
                backward_frontier.append(new_path)
                backward_visited[neighbor] = new_path  # Track full path
                
                if neighbor in forward_visited:
                    # Combine paths, ensuring the common node is not duplicated
                    return forward_visited[neighbor] + backward_visited[neighbor][::-1][1:]

    return []  # Return empty if no path is found
#tab completion function
# def completer(text, state):
#     matches = [place for place in places if place.lower().startswith(text.lower())]
#     if state < len(matches):              does not work in spyder
#         return matches[state]
#     else:
#         return None

global THE_path
global G
G = ox.graph_from_place("Karbala, Iraq", network_type="drive", simplify=False)
#print("Edges:", G.edges(data=True))

#                  (latitude, longitude)
global places
places = {
    "Al Askary" : [32.650818, 43.9781],
    "Soumer" : [32.6481, 43.9883],
    "Al Qadisia" : [32.643024, 43.984644],
    "Al Arabi" : [32.633728,43.967146],
    "The Turkish Hospital" : [32.644308, 43.969860],
    "The Worker neighborhood" : [32.629990, 43.980509],
    "Al Mujtaba" : [32.636777,43.991382],
    "Al Ayobeien" : [32.628189, 43.992965],
    "AL Mualmeen" : [32.617093,44.002175],
    "The Employeds neighborhood" : [32.605163, 44.004450],
    "Al Salam" : [32.587826,43.991822],
    "Al Tahadie" : [32.594199, 44.004600],
    "The Family neighborhood" : [32.584339, 44.032735],
    "Al Iskan" : [32.5940417, 44.026728],
    "Al Hussain" : [32.597728, 44.017239],
    "Al Nasar" : [32.577294,44.006423],
    "The Doctors neighborhood" : [32.584790, 43.999386],
    "Al Senaaie" : [32.577215, 44.047531],
    "Al Chaier" : [32.507188, 44.043860],
    "Al Abbas" : [32.636017, 44.049627],
    "Al Hur" : [32.650316, 43.985929],
    "The Small Hur" : [32.653217,43.999199],


    "Imam Hussain Shrine": [32.6160, 44.0316],
    "Al Abbas Shrine": [32.6135, 44.0361],
    "Karbala University": [32.601329, 44.090067],
    "Karbala Stadium" : [32.565359, 44.004452],
    

    
}
#readline.set_completer(completer)
# readline.parse_and_bind("tab: complete")          require readline lib
places_completer = WordCompleter(places.keys(), ignore_case=True)
start = prompt("Where do you start? ", completer=places_completer)
stop = prompt("Where do you stop? ", completer=places_completer)

# start and end 
start_point = places[start]
end_point = places[stop]

# Find the nearest nodes to the start and end points
start_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])
end_node = ox.distance.nearest_nodes(G, end_point[1], end_point[0])
#
choice_completer = WordCompleter(["yes" , "no"] , ignore_case=True)
choice = prompt("Map Solved? yes or no? " , completer=choice_completer)
if choice == "yes":
    chosen_path = ""
    while chosen_path == "":
        path_functions = {
        'BFS': BFS,
        'DFS': DFS,
        'UCS': UCS,
        'DLS': DLS,
        'IDDFS': IDDFS,
        'BDS' : BDS,
        'SOLVE': nx.shortest_path(G, start_node, end_node, weight='length') }
        chosen_path_completer = WordCompleter(path_functions.keys() , ignore_case=True)
        chosen_path = prompt(" please choose the path that you want. we have \n BFS \n DFS \n UCS \n DLS \n IDDFS \n BDS \n andddddddd WE have THE SOLUTION!!!!! just type solve:   " , completer = chosen_path_completer)
    i = 0
    if chosen_path in path_functions:
        #bfs_paths = nx.single_source_shortest_path(G, start_node)
        #bfs_path = bfs_paths.get(end_node, None)
        if chosen_path == 'SOLVE':
            THE_path = nx.shortest_path(G, start_node, end_node, weight='length')
        elif (chosen_path == 'DLS'):
            depth_limit = input("Please enter the depth limit: ")
            THE_path = path_functions[chosen_path](G , start_node , end_node , int(depth_limit))
        elif (chosen_path == 'IDDFS'):
            depth_limit = input("Please enter the depth limit: ")
            THE_path = path_functions[chosen_path](G , start_node , end_node , int(depth_limit))
       # elif (chosen_path != path_functions['DLS'] and chosen_path != path_functions['IDDFS']):
            
        else:
            THE_path = path_functions[chosen_path](G , start_node , end_node)
            
        # Find the path section
        # shortest_path = nx.shortest_path(G, start_node, end_node, weight="length")
        if THE_path:
            #path_distance = nx.shortest_path_length(G, start_node, end_node, weight="length")
            path_distance = path_distance_calc( G , THE_path)
            #print(f"the length of path is {len(THE_path)} and the length of the graph is is {len(G)}")   # for debugging fuck
            print(f"Distance from start to end node: {path_distance:.2f} meters")
            # Plot the route
#idk really know why this dont work ↓
#fig, ax = ox.plot_graph_route(G, bfs_path, route_linewidth=2, node_size=0, bgcolor="white")
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the graph of the road network onto the axis
ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor="white", show=False, close=False)

# Plot the route on top of the graph
if choice == 'yes':
    ox.plot_graph_route(G, THE_path , route_linewidth=1, node_size=-5, ax=ax, route_alpha=0.7, show=False, close=False)

# Add markers for each known place
for place, (lat, lon) in places.items():
    # Find the nearest node to each known place
    node = ox.distance.nearest_nodes(G, lon, lat)
    x, y = G.nodes[node]['x'], G.nodes[node]['y']
    
    # Plot a marker at each known place
    ax.plot(x, y, 'o', color='black', markersize=5, zorder=5)
    ax.text(x + 0.001, y, place, fontsize=7, color='darkred', fontstyle = "italic" , fontfamily = 'serif' ,  ha='left', zorder=9)

# Show the plot with all layers combined
# After plotting the graph and path
if choice == 'yes':
    arrow = prompt("Add arrow indicator for the way(messy) : yes or no? ", completer=choice_completer)
    if arrow == 'yes':
        for i in range(len(THE_path) - 1):
            u = THE_path[i]
            v = THE_path[i + 1]
            
            # Get coordinates for the start and end node of the edge
            x_start, y_start = G.nodes[u]['x'], G.nodes[u]['y']
            x_end, y_end = G.nodes[v]['x'], G.nodes[v]['y']
        
        # Add an arrow using annotate
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=0.30))

plt.show()
if choice == 'yes' and not THE_path :
    print(f"No path found from start to end node using {chosen_path}.")   
