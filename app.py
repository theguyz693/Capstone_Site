from flask import Flask, render_template, request, jsonify
from graph import ALGORITHMS, graph
import time

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    comparison_results = []
    available_nodes = sorted(graph.keys())
    
    if request.method == "POST":
        start = request.form.get("start", "").strip()
        end = request.form.get("end", "").strip()
        algo = request.form.get("algorithm")
        compare_mode = request.form.get("compare_mode") == "on"
        
        if start and end:
            if compare_mode:
                # Compare all algorithms
                for algo_key, algo_info in ALGORITHMS.items():
                    try:
                        start_time = time.time()
                        
                        if algo_key == 'dynamic':
                            path, cost, nodes_explored = algo_info['function'](start, end, set())
                        elif algo_key == 'mcsp':
                            path, cost, nodes_explored = algo_info['function'](start, end)
                        else:
                            path, cost, nodes_explored = algo_info['function'](start, end)
                        
                        execution_time = (time.time() - start_time) * 1000  # ms
                        
                        comparison_results.append({
                            'name': algo_info['name'],
                            'key': algo_key,
                            'path': ' → '.join(path) if path else 'No path found',
                            'cost': f"{cost:.2f}" if cost is not None else "N/A",
                            'nodes_explored': nodes_explored,
                            'execution_time': f"{execution_time:.4f}",
                            'path_length': len(path) if path else 0,
                            'pros': algo_info['pros'],
                            'cons': algo_info['cons'],
                            'complexity': algo_info['complexity'],
                            'best_for': algo_info['best_for']
                        })
                    except Exception as e:
                        comparison_results.append({
                            'name': algo_info['name'],
                            'key': algo_key,
                            'path': f'Error: {str(e)}',
                            'cost': 'N/A',
                            'nodes_explored': 0,
                            'execution_time': '0',
                            'path_length': 0,
                            'pros': algo_info['pros'],
                            'cons': algo_info['cons'],
                            'complexity': algo_info['complexity'],
                            'best_for': algo_info['best_for']
                        })
                
                # Sort by execution time
                comparison_results.sort(key=lambda x: float(x['execution_time']))
                
            else:
                # Single algorithm
                if algo and algo in ALGORITHMS:
                    algo_info = ALGORITHMS[algo]
                    try:
                        start_time = time.time()
                        
                        if algo == 'dynamic':
                            path, cost, nodes_explored = algo_info['function'](start, end, set())
                        elif algo == 'mcsp':
                            path, cost, nodes_explored = algo_info['function'](start, end)
                        else:
                            path, cost, nodes_explored = algo_info['function'](start, end)
                        
                        execution_time = (time.time() - start_time) * 1000
                        
                        result = {
                            'name': algo_info['name'],
                            'path': ' → '.join(path) if path else 'No path found',
                            'cost': f"{cost:.2f}" if cost is not None else "N/A",
                            'nodes_explored': nodes_explored,
                            'execution_time': f"{execution_time:.4f}",
                            'path_length': len(path) if path else 0,
                            'pros': algo_info['pros'],
                            'cons': algo_info['cons'],
                            'complexity': algo_info['complexity'],
                            'best_for': algo_info['best_for']
                        }
                    except Exception as e:
                        result = {
                            'name': algo_info['name'],
                            'path': f'Error: {str(e)}',
                            'cost': 'N/A',
                            'nodes_explored': 0,
                            'execution_time': '0',
                            'path_length': 0,
                            'pros': algo_info['pros'],
                            'cons': algo_info['cons'],
                            'complexity': algo_info['complexity'],
                            'best_for': algo_info['best_for']
                        }
    
    return render_template(
        "index.html",
        result=result,
        comparison_results=comparison_results,
        algorithms=ALGORITHMS,
        available_nodes=available_nodes,
        graph_data=graph
    )

@app.route("/api/graph")
def get_graph():
    """API endpoint to get graph data"""
    nodes = []
    edges = []
    
    for node in graph.keys():
        floor = 0
        if node.startswith('R1') or node == 'S1':
            floor = 0
        elif node.startswith('R2') or node in ['S2', 'S3']:
            floor = 1
        elif node.startswith('R3') or node == 'S4':
            floor = 2
        
        nodes.append({
            "id": node,
            "label": node,
            "floor": floor,
            "type": "stairs" if node.startswith('S') else "room"
        })
    
    edge_set = set()
    for source, targets in graph.items():
        for target, weight in targets.items():
            edge_key = tuple(sorted([source, target]))
            if edge_key not in edge_set:
                edges.append({
                    "from": source,
                    "to": target,
                    "weight": weight
                })
                edge_set.add(edge_key)
    
    return jsonify({"nodes": nodes, "edges": edges})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
