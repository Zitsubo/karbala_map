from flask import Flask, render_template, request, render_template_string
import osmnx as ox
import networkx as nx
import folium
import json
# made py ZITSUBO 2024/11/13
app = Flask(__name__)

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

            m = folium.Map(location=[start_point[0], start_point[1]], zoom_start=13, tiles="CartoDB positron")

            # route plot
            route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in THE_path]
            folium.PolyLine(locations=route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

            for place, (lat, lon) in places.items():
                is_start = place == start and start != "custom"
                is_stop = place == stop and stop != "custom"

                if is_start:
                    folium.Marker([lat, lon], popup=place, tooltip=place, icon=folium.Icon(color="green")).add_to(m)
                elif is_stop:
                    folium.Marker([lat, lon], popup=place, tooltip=place, icon=folium.Icon(color="red")).add_to(m)
                else:
                    folium.Marker([lat, lon], popup=place, tooltip=place).add_to(m)

            if start == "custom":
                folium.Marker(start_point, popup="Custom Start", tooltip="Custom Start", icon=folium.Icon(color="green")).add_to(m)
            if stop == "custom":
                folium.Marker(end_point, popup="Custom Stop", tooltip="Custom Stop", icon=folium.Icon(color="red")).add_to(m)

            html_text = f"""
            <div style="position: fixed;
                        bottom: 50px; left: 50px; width: 350px; height: 160px;
                        background-color: white; border:2px solid black; z-index:9999; font-size:16px;
                        font-family: serif; font-weight: bold; padding: 10px;">
                <p>Start is {start}</p>
                <p>Goal is {stop}</p>
                <p>Distance is {path_distance:.2f}m</p>
                <p>ETA is {time}</p>
            </div>
            """
            m.get_root().html.add_child(folium.Element(html_text))

            map_html = m.get_root().render()
            return render_template("index.html", places=places.keys(), places_json=places_json, map_html=map_html, start=start, stop=stop, speed=speed)
        else:
            error_msg = f"No path found from {start} to {stop}."
            return render_template("index.html", places=places.keys(), places_json=places_json, error=error_msg)

    return render_template("index.html", places=places.keys(), places_json=places_json)

if __name__ == "__main__":
    app.run(debug=True)
