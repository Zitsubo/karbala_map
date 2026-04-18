from flask import Flask, render_template, request, render_template_string
import osmnx as ox
import networkx as nx
import folium
import heapq
import os

app = Flask(__name__)

# Global variables setup
global depth_of_the_search
depth_of_the_search = 0
limit = 0 
max_depth = 0

print("Loading graph...")
G = ox.graph_from_place("Karbala, Iraq", network_type="drive", simplify=False)
G = ox.simplification.simplify_graph(G)
print("Graph loaded.")

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

algorithms = ["BFS" , "DFS" , "UCS" , "Greedy" , "A*" , "DLS" , "IDDFS" , "Magic Algorithm"]

def DLS(graph, start, goal , limit):
    def recursive_dls(node, goal, path, depth):
        if node in path:
            return None
        path.append(node)
        if node == goal:
            return path
        if depth >= limit:
            path.pop()
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

def TheAlgorithms(algorithm, graph, start, goal, limit_val=0, max_depth_val=0):
    global depth_of_the_search
    depth_of_the_search = 0
    if algorithm == "DLS":
        global limit
        limit = limit_val
        return DLS(graph , start , goal , limit)
    elif algorithm == "IDDFS":
        global max_depth
        max_depth = max_depth_val
        return IDDFS(graph , start , goal , max_depth)
    if algorithm == "Magic Algorithm":
        return (nx.shortest_path(G, start, goal, weight='length'))
    elif algorithm in ["BFS", "DFS"]:
        visited = []
        queue = [[start]]
    elif algorithm in ["UCS", "A*", "Greedy"]:
        queue = []
        heapq.heappush(queue, (0, [start]))
        visited = set()

    if algorithm in ["UCS", "A*", "Greedy"]:
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
    return None

def path_distance_calc(graph, path):
    total_distance = 0
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i] ,path[i + 1])
        
        if edge_data:
            for _, data in edge_data.items():
                total_distance += data.get('length', 0) 
    return total_distance

def ETA(graph, path, speed):
    total_distance = path_distance_calc(graph , path)
    speed = float(speed)
    if speed <= 0: return "00:00:00"
    time_by_seconds = (total_distance / (speed * 1000)) * 3600
    hours = int((time_by_seconds / 3600))
    minutes = int((time_by_seconds % 3600) / 60)
    seconds = int(time_by_seconds % 60)
    time = f"{hours:02}:{minutes:02}:{seconds:02}"
    return time

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start = request.form.get("start")
        stop = request.form.get("stop")
        algorithm = request.form.get("algorithm")
        speed = float(request.form.get("speed") or 50)
        limit_val = int(request.form.get("limit") or 0)
        max_depth_val = int(request.form.get("max_depth") or 0)

        start_point = places[start]
        end_point = places[stop]
        start_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])
        end_node = ox.distance.nearest_nodes(G, end_point[1], end_point[0])

        THE_path = TheAlgorithms(algorithm , G , start_node , end_node, limit_val, max_depth_val)

        if THE_path:
            path_distance = path_distance_calc(G , THE_path)
            time = ETA(G, THE_path, speed)

            m = folium.Map(location=[start_point[0], start_point[1]], zoom_start=13, tiles="CartoDB positron")

            # route plot
            route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in THE_path]
            folium.PolyLine(locations=route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

            for place, (lat, lon) in places.items():
                if place == start:
                    folium.Marker([lat, lon], popup=place, tooltip=place, icon=folium.Icon(color="green")).add_to(m)
                elif place == stop:
                    folium.Marker([lat, lon], popup=place, tooltip=place, icon=folium.Icon(color="red")).add_to(m)
                else:
                    folium.Marker([lat, lon], popup=place, tooltip=place).add_to(m)

            html_text = f"""
            <div style="position: fixed;
                        bottom: 50px; left: 50px; width: 350px; height: 160px;
                        background-color: white; border:2px solid black; z-index:9999; font-size:16px;
                        font-family: serif; font-weight: bold; padding: 10px;">
                <p>Start is {start}</p>
                <p>Goal is {stop}</p>
                <p>Distance is {path_distance:.2f}m at depth {depth_of_the_search}</p>
                <p>ETA is {time}</p>
            </div>
            """
            m.get_root().html.add_child(folium.Element(html_text))

            map_html = m.get_root().render()
            return render_template("index.html", places=places.keys(), algorithms=algorithms, map_html=map_html, start=start, stop=stop, algorithm=algorithm, speed=speed)
        else:
            error_msg = f"No path found from {start} to {stop} using {algorithm}."
            return render_template("index.html", places=places.keys(), algorithms=algorithms, error=error_msg)

    return render_template("index.html", places=places.keys(), algorithms=algorithms)

if __name__ == "__main__":
    app.run(debug=True)
