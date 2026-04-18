import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import heapq
import os
# made py ZITSUBO 2024/11/13
# my search algorithms for whatever reason
global depth_of_the_search
depth_of_the_search = 0
limit = 0 
max_depth = 0
def TheAlgorithms(algorithm, graph, start, goal): #BFS and DFS and UCS and Greedy and Astar and the other two
    global depth_of_the_search
    if algorithm == "DLS":
        global limit
        limit = int(input("The limit of the search: "))
        DLS(graph , start , goal , limit)
    elif algorithm == "IDDFS":
        global max_depth
        max_depth = int(input("The Max depth of the search: "))
        IDDFS(graph , start , goal , max_depth)
    if algorithm == "Magic Algorithm":
        return (nx.shortest_path(G, start_node, end_node, weight='length'))
    elif algorithm in ["BFS", "DFS"]:
        visited = []
        queue = [[start]]
    elif algorithm in ["UCS", "A*", "Greedy"]:
        queue = []
        heapq.heappush(queue, (0, [start]))
        visited = set()
        while queue:
            current_cost, path = heapq.heappop(queue)
            node = path[-1]
            if node in visited:
                continue
            visited.add(node)
            if node == goal:
                return path
            depth_of_the_search+=1
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    edge_data = graph.get_edge_data(node, neighbor)
                    cost = edge_data[0].get('length', 0)
                    if algorithm == "A*":
                        heuristic = ox.distance.euclidean(graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'],
                                                         graph.nodes[goal]['y'], graph.nodes[goal]['x'])
                        new_cost = current_cost + cost + heuristic
                    elif algorithm == "Greedy":
                        new_cost = ox.distance.euclidean(graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'],
                                                         graph.nodes[goal]['y'], graph.nodes[goal]['x'])
                    else:
                        new_cost = current_cost + cost
                    new_path = path + [neighbor]
                    heapq.heappush(queue, (new_cost, new_path))
    if algorithm in ["BFS", "DFS"]:
        while queue:
            if algorithm == "BFS":
                path = queue.pop(0)
                node = path[-1]
            elif algorithm == "DFS":
                path = queue.pop()
                node = path[-1]
            if node not in visited:
                visited.append(node)
            else:
                continue
            if node == goal:
                return path
            depth_of_the_search+=1
            for neighbor in graph[node]:
                new_path = path.copy()
                new_path.append(neighbor)
                queue.append(new_path)

def path_distance_calc(graph, path):
    total_distance = 0
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i] ,path[i + 1])
        
        if edge_data:
            for _, data in edge_data.items():
                total_distance += data.get('length', 0) 
    print(f"Distance from {start} to {stop} is : {total_distance:.2f}m at depth {depth_of_the_search}")
    return total_distance
         
def DLS(graph, start, goal , limit):
    def recursive_dls(node, goal, path, depth):
        if node in path:
            return
        path.append(node)
        if node == goal:
            return path
        if depth >= limit:
            return None

        for neighbor in graph[node]:
            if neighbor not in path:
                result = recursive_dls(neighbor, goal, path, depth + 1)
                if result is not None:
                    return result 
        path.pop()
        return None
    return recursive_dls(start, goal, [], 0)

def IDDFS(graph, start, goal , max_depth):
    for depth in range(max_depth + 1):
        path = DLS(graph, start, goal, depth)
        if path is not None:
            return path
    return None


def ETA(graph , path):
    total_distance = path_distance_calc(graph , path)
    speed = ""
    while speed == "":
        speed = input("Whats your average speed? by KM:  ")
    speed = float(speed)
    time_by_seconds = (total_distance / (speed * 1000)) * 3600
    hours = int((time_by_seconds / 3600))
    minutes = int((time_by_seconds % 3600) / 60)
    seconds = int(time_by_seconds % 60)
    #time = round(time)
    time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"The Eastimated time to arrive is {hours:02}:{minutes:02}:{seconds:02}")
    return time


global THE_path
global start , stop
global G
G = ox.graph_from_place("Karbala, Iraq", network_type="drive", simplify=False)
G = ox.simplification.simplify_graph(G)

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
    "Al Chaier" : [32.597420, 44.041025],
    "Al Abbas" : [32.636017, 44.049627],
    "Al Hur" : [32.650316, 43.985929],
    "The Small Hur" : [32.653217,43.999199],
    "Imam Hussain Shrine": [32.6160, 44.0316],
    "Al Abbas Shrine": [32.6135, 44.0361],
    "Karbala University": [32.601329, 44.090067],
    "Karbala Stadium" : [32.565359, 44.004452],
}
algorithms = {"BFS" , "DFS" , "UCS" , "Greedy" , "A*" , "DLS" , "IDDFS" , "Magic Algorithm"}
algorithms_completer = WordCompleter(algorithms , ignore_case=True)
places_completer = WordCompleter(places.keys(), ignore_case=True)
choice_completer = WordCompleter(["yes" , "no"] , ignore_case=True)
start = ""
stop = ""
algorithm = ""

while start == "":
    start = prompt("Where do you start? ", completer=places_completer)
while stop == "":
    stop = prompt("Where do you stop? ", completer=places_completer)
while algorithm == "": 
    algorithm = prompt("What Algorithm? " , completer=algorithms_completer)

start_point = places[start]
end_point = places[stop]
start_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])
end_node = ox.distance.nearest_nodes(G, end_point[1], end_point[0])

THE_path = TheAlgorithms(algorithm , G , start_node , end_node)
global time
if THE_path:
    path_distance = path_distance_calc( G , THE_path)
    time = ETA(G , THE_path)
    choice = ""
    while choice == "":
        choice = prompt("Draw The Map? " , completer=choice_completer)
    if choice == "yes":
        fig, ax = plt.subplots(figsize=(12, 12))
        ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.5, bgcolor="white", show=False, close=False)
                                                # route plot
        ox.plot_graph_route(G, THE_path , route_linewidth=1, node_size=-5, ax=ax, route_alpha=0.7, show=False, close=False)

        for place, (lat, lon) in places.items():
            node = ox.distance.nearest_nodes(G, lon, lat)
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            ax.plot(x, y, 'o', color='black', markersize=5, zorder=5)
            ax.text(x + 0.001, y, place, fontsize=10, color='darkred', fontstyle = "italic" , fontfamily = 'serif' ,  ha='left', zorder=9)

        for i in range(len(THE_path) - 1):
            u = THE_path[i]
            v = THE_path[i + 1]
            x_start, y_start = G.nodes[u]['x'], G.nodes[u]['y']
            x_end, y_end = G.nodes[v]['x'], G.nodes[v]['y']
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start), arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=0.30))
        plt.figtext(
        0.05, 0.95,  # text position
        f"Start is {start}\nGoal is {stop}\nDistance is {path_distance:.2f}m at depth {depth_of_the_search}\n ETA is {time}\n ",
        fontsize=20,  
        color="Black",
        fontweight="bold",
        fontfamily="serif",
        ha="left",
        va="top"
    )
        plt.show()
        plt.savefig("Karbala-map.png" , format = "png",
        dpi = 600 )
    elif choice == "no":
        if os.name == 'nt':
            os.system("cls")
        elif os.name == 'posix':
            os.system("clear")
        path_distance_calc( G , THE_path)
        print(f"The Eastimated time to arrive is {time}")
if not THE_path and algorithm != "DLS" and algorithm != "IDDFS":
    print(f"No path found from {start} to {stop} using {algorithm}.")
if not THE_path and algorithm == "DLS" or algorithm == "IDDFS":
    if algorithm == "DLS":
        print(f"No path found from {start} to {stop} using {algorithm} at depth {limit}.")
    if algorithm == "IDDFS":
        print(f"No path found from {start} to {stop} using {algorithm} at depth {max_depth}.")
    exit(1)
