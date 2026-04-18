from flask import Flask, render_template, request, render_template_string
import osmnx as ox
import networkx as nx
import json
# made py ZITSUBO 2024/11/13
# my search algorithms for whatever reason

app = Flask(__name__)

print("Loading graph...")
G = ox.graph_from_place("Karbala, Iraq", network_type="drive", simplify=False)
G = ox.simplification.simplify_graph(G)
print("Graph loaded.")

places = {
    "حي العسكري" : [32.650818, 43.9781],
    "سومر" : [32.6481, 43.9883],
    "القادسية" : [32.643024, 43.984644],
    "حي العربي" : [32.633728,43.967146],
    "المستشفى التركي" : [32.644308, 43.969860],
    "حي العامل" : [32.629990, 43.980509],
    "المجتبى" : [32.636777,43.991382],
    "الايوبيين" : [32.628189, 43.992965],
    "حي المعلمين" : [32.617093,44.002175],
    "حي الموظفين" : [32.605163, 44.004450],
    "السلام" : [32.587826,43.991822],
    "التحدي" : [32.594199, 44.004600],
    "حي الاسرة" : [32.584339, 44.032735],
    "الاسكان" : [32.5940417, 44.026728],
    "حي الحسين" : [32.597728, 44.017239],
    "النصر" : [32.577294,44.006423],
    "حي الاطباء" : [32.584790, 43.999386],
    "الحي الصناعي" : [32.577215, 44.047531],
    "الجاير" : [32.597420, 44.041025],
    "حي العباس" : [32.636017, 44.049627],
    "الحر" : [32.650316, 43.985929],
    "الحر الصغير" : [32.653217,43.999199],
    "مرقد الامام الحسين": [32.6160, 44.0316],
    "مرقد الامام العباس": [32.6135, 44.0361],
    "جامعة كربلاء": [32.601329, 44.090067],
    "ملعب كربلاء" : [32.565359, 44.004452],
}

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
    places_json = json.dumps(places)
    if request.method == "POST":
        start = request.form.get("start")
        stop = request.form.get("stop")
        speed = float(request.form.get("speed") or 50)

        start_lat = float(request.form.get("start_lat"))
        start_lon = float(request.form.get("start_lon"))
        stop_lat = float(request.form.get("stop_lat"))
        stop_lon = float(request.form.get("stop_lon"))

        start_point = [start_lat, start_lon]
        end_point = [stop_lat, stop_lon]

        start_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])
        end_node = ox.distance.nearest_nodes(G, end_point[1], end_point[0])

        try:
            THE_path = nx.shortest_path(G, start_node, end_node, weight='length')
        except nx.NetworkXNoPath:
            THE_path = None

        if THE_path:
            path_distance = path_distance_calc(G , THE_path)
            time = ETA(G, THE_path, speed)

            route_coords = [[G.nodes[node]['y'], G.nodes[node]['x']] for node in THE_path]
            route_coords_json = json.dumps(route_coords)

            return render_template(
                "index.html",
                places=places.keys(),
                places_json=places_json,
                route_coords_json=route_coords_json,
                start=start,
                stop=stop,
                speed=speed,
                path_distance=path_distance,
                time=time
            )
        else:
            error_msg = f"No path found from {start} to {stop}."
            return render_template("index.html", places=places.keys(), places_json=places_json, error=error_msg)

    return render_template("index.html", places=places.keys(), places_json=places_json)

if __name__ == "__main__":
    app.run(debug=True)
