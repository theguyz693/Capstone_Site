import heapq
from collections import deque
import math

# Enhanced 2-Floor Graph Structure (Based on uploaded image)
# Ground Floor: R11-R15 connected via S1
# First Floor: R21-R24 connected via S2 and S3
# Second Floor: R31-R35 connected via S4
# Stairs: S1 (ground-first), S2, S3 (first floor connectors), S4 (first-second)

graph = {
    # Ground Floor (Floor 0)
    "R11": {"S1": 3, "R12": 5},
    "R12": {"R11": 5, "R13": 4},
    "R13": {"R12": 4, "R14": 4},
    "R14": {"R13": 4, "R15": 3},
    "R15": {"R14": 3, "S1": 3},
    
    # Stairs S1 (Ground to First Floor connector)
    "S1": {"R11": 3, "R15": 3, "S2": 8},  # 8 units for vertical movement
    
    # First Floor - Main corridor
    "S2": {"S1": 8, "R21": 3, "S4": 10, "S3": 6},
    "R21": {"S2": 3, "S3": 4},
    "S3": {"R21": 4, "R22": 3, "S4": 5, "S2": 6},
    "R22": {"S3": 3, "R23": 4},
    "R23": {"R22": 4, "R24": 4},
    "R24": {"R23": 4},
    
    # Stairs S4 (First to Second Floor connector)
    "S4": {"S2": 10, "S3": 5, "R33": 3, "R34": 3},
    
    # Second Floor
    "R31": {"R32": 4},
    "R32": {"R31": 4, "R33": 4},
    "R33": {"R32": 4, "S4": 3},
    "R34": {"S4": 3, "R35": 4},
    "R35": {"R34": 4}
}

# Node positions for heuristic (based on your graph layout)
node_positions = {
    # Ground Floor (bottom layer)
    "R11": (4, 0),
    "R12": (5, 1),
    "S1": (3, 1),
    "R15": (2, 1),
    "R14": (2, 2),
    "R13": (3, 2),
    
    # First Floor (middle layer)
    "S2": (2, 4),
    "R21": (3, 4),
    "S3": (4, 4),
    "R22": (5, 4),
    "R23": (6, 4),
    "R24": (7, 4),
    
    # Second Floor (top layer)
    "R31": (1, 7),
    "R32": (2, 7),
    "R33": (3, 7),
    "S4": (4, 7),
    "R34": (5, 7),
    "R35": (6, 7)
}

def heuristic(node, goal):
    """Calculate Euclidean distance heuristic for A*"""
    if node not in node_positions or goal not in node_positions:
        return 0
    x1, y1 = node_positions[node]
    x2, y2 = node_positions[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# 1. BREADTH-FIRST SEARCH (BFS)
def bfs(start, goal):
    """BFS: Explores level by level, finds path with minimum hops"""
    if start not in graph or goal not in graph:
        return None, None, 0
    
    queue = deque([[start]])
    visited = set()
    nodes_explored = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        nodes_explored += 1
        
        if node == goal:
            cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            return path, cost, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, {}):
                if neighbor not in visited:
                    queue.append(path + [neighbor])
    
    return None, None, nodes_explored


# 2. DEPTH-FIRST SEARCH (DFS)
def dfs(start, goal):
    """DFS: Explores deeply, may not find optimal path"""
    if start not in graph or goal not in graph:
        return None, None, 0
    
    stack = [(start, [start])]
    visited = set()
    nodes_explored = 0

    while stack:
        node, path = stack.pop()
        nodes_explored += 1
        
        if node == goal:
            cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            return path, cost, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, {}):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    
    return None, None, nodes_explored


# 3. DIJKSTRA'S ALGORITHM
def dijkstra(start, goal):
    """Dijkstra: Optimal weighted shortest path, guaranteed optimal"""
    if start not in graph or goal not in graph:
        return None, None, 0
    
    pq = [(0, start, [])]
    visited = set()
    nodes_explored = 0

    while pq:
        cost, node, path = heapq.heappop(pq)
        nodes_explored += 1
        
        if node == goal:
            return path + [node], cost, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + weight, neighbor, path + [node]))
    
    return None, None, nodes_explored


# 4. A* ALGORITHM
def astar(start, goal):
    """A*: Uses heuristic for faster pathfinding"""
    if start not in graph or goal not in graph:
        return None, None, 0
    
    pq = [(heuristic(start, goal), 0, start, [])]
    visited = set()
    nodes_explored = 0

    while pq:
        f_score, g_score, node, path = heapq.heappop(pq)
        nodes_explored += 1
        
        if node == goal:
            return path + [node], g_score, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph.get(node, {}).items():
                if neighbor not in visited:
                    new_g = g_score + weight
                    new_f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [node]))
    
    return None, None, nodes_explored


# 5. BELLMAN-FORD ALGORITHM
def bellman_ford(start, goal):
    """Bellman-Ford: Handles negative weights, detects cycles"""
    if start not in graph or goal not in graph:
        return None, None, 0
    
    nodes = list(graph.keys())
    distance = {node: float('inf') for node in nodes}
    predecessor = {node: None for node in nodes}
    distance[start] = 0
    nodes_explored = 0

    for _ in range(len(nodes) - 1):
        for node in nodes:
            for neighbor, weight in graph.get(node, {}).items():
                nodes_explored += 1
                if distance[node] + weight < distance[neighbor]:
                    distance[neighbor] = distance[node] + weight
                    predecessor[neighbor] = node

    if distance[goal] == float('inf'):
        return None, None, nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.insert(0, current)
        current = predecessor[current]
    
    return path, distance[goal], nodes_explored


# 6. FLOYD-WARSHALL ALGORITHM
def floyd_warshall(start, goal):
    """Floyd-Warshall: All-pairs shortest path"""
    nodes = list(graph.keys())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    dist = [[float('inf')] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for node in nodes:
        for neighbor, weight in graph.get(node, {}).items():
            i, j = node_index[node], node_index[neighbor]
            dist[i][j] = weight
            next_node[i][j] = neighbor
    
    nodes_explored = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                nodes_explored += 1
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    if start not in node_index or goal not in node_index:
        return None, None, nodes_explored
    
    start_idx, goal_idx = node_index[start], node_index[goal]
    if dist[start_idx][goal_idx] == float('inf'):
        return None, None, nodes_explored
    
    path = [start]
    current = start_idx
    while current != goal_idx:
        current = node_index[next_node[current][goal_idx]]
        path.append(nodes[current])
    
    return path, dist[start_idx][goal_idx], nodes_explored


# 7. JOHNSON'S ALGORITHM
def johnson(start, goal):
    """Johnson's: Efficient all-pairs for sparse graphs"""
    nodes = list(graph.keys())
    extended_graph = {node: dict(graph[node]) for node in graph}
    extended_graph['__temp__'] = {node: 0 for node in nodes}
    
    distance = {node: float('inf') for node in extended_graph}
    distance['__temp__'] = 0
    nodes_explored = 0
    
    for _ in range(len(extended_graph) - 1):
        for node in extended_graph:
            for neighbor, weight in extended_graph[node].items():
                nodes_explored += 1
                if distance[node] + weight < distance[neighbor]:
                    distance[neighbor] = distance[node] + weight
    
    h = distance
    reweighted_graph = {}
    for node in graph:
        reweighted_graph[node] = {}
        for neighbor, weight in graph[node].items():
            reweighted_graph[node][neighbor] = weight + h[node] - h[neighbor]
    
    pq = [(0, start, [])]
    visited = set()
    
    while pq:
        cost, node, path = heapq.heappop(pq)
        nodes_explored += 1
        
        if node == goal:
            original_cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)))
            return path + [node], original_cost, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor, weight in reweighted_graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + weight, neighbor, path + [node]))
    
    return None, None, nodes_explored


# 8. MULTI-CRITERIA SHORTEST PATH (MCSP)
def mcsp(start, goal, criteria_weights=None):
    """MCSP: Considers distance, stairs penalty, and congestion"""
    if criteria_weights is None:
        criteria_weights = {'distance': 0.5, 'stairs': 0.3, 'congestion': 0.2}
    
    congestion = {
        "S1": 0.7, "S2": 0.8, "S3": 0.6, "S4": 0.5,
        "R21": 0.4, "R22": 0.3, "R23": 0.2
    }
    
    stairs_nodes = {"S1", "S2", "S3", "S4"}
    
    if start not in graph or goal not in graph:
        return None, None, 0
    
    pq = [(0, start, [])]
    visited = set()
    nodes_explored = 0

    while pq:
        total_cost, node, path = heapq.heappop(pq)
        nodes_explored += 1
        
        if node == goal:
            return path + [node], total_cost, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor, distance in graph.get(node, {}).items():
                if neighbor not in visited:
                    cong_factor = congestion.get(neighbor, 0.1)
                    stairs_penalty = 2.0 if neighbor in stairs_nodes else 0
                    
                    combined_cost = (
                        criteria_weights['distance'] * distance +
                        criteria_weights['stairs'] * stairs_penalty +
                        criteria_weights['congestion'] * cong_factor * distance
                    )
                    heapq.heappush(pq, (total_cost + combined_cost, neighbor, path + [node]))
    
    return None, None, nodes_explored


# 9. DYNAMIC SHORTEST PATH
def dynamic_shortest_path(start, goal, blocked_nodes=None):
    """Dynamic: Re-routes around obstacles"""
    if blocked_nodes is None:
        blocked_nodes = set()
    
    if start not in graph or goal not in graph:
        return None, None, 0
    
    temp_graph = {}
    for node in graph:
        if node not in blocked_nodes:
            temp_graph[node] = {
                neighbor: weight 
                for neighbor, weight in graph[node].items() 
                if neighbor not in blocked_nodes
            }
    
    pq = [(heuristic(start, goal), 0, start, [])]
    visited = set()
    nodes_explored = 0

    while pq:
        f_score, g_score, node, path = heapq.heappop(pq)
        nodes_explored += 1
        
        if node == goal:
            return path + [node], g_score, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor, weight in temp_graph.get(node, {}).items():
                if neighbor not in visited:
                    new_g = g_score + weight
                    new_f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [node]))
    
    return None, None, nodes_explored


# 10. CONTRACTION HIERARCHIES
def contraction_hierarchies(start, goal):
    """Contraction Hierarchies: Fast queries via node importance"""
    node_importance = {}
    for node in graph:
        if node.startswith('S'):
            node_importance[node] = len(graph[node]) * 2
        else:
            node_importance[node] = len(graph[node])
    
    sorted_nodes = sorted(graph.keys(), key=lambda n: node_importance[n])
    node_level = {node: i for i, node in enumerate(sorted_nodes)}
    
    if start not in graph or goal not in graph:
        return None, None, 0
    
    forward_pq = [(0, start, [])]
    forward_visited = {}
    backward_pq = [(0, goal, [])]
    backward_visited = {}
    
    best_path = None
    best_cost = float('inf')
    nodes_explored = 0
    
    while forward_pq or backward_pq:
        if forward_pq:
            cost, node, path = heapq.heappop(forward_pq)
            nodes_explored += 1
            
            if node not in forward_visited or cost < forward_visited[node][0]:
                forward_visited[node] = (cost, path + [node])
                
                if node in backward_visited:
                    total_cost = cost + backward_visited[node][0]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path + [node] + backward_visited[node][1][::-1][1:]
                
                for neighbor, weight in graph.get(node, {}).items():
                    if node_level[neighbor] >= node_level[node]:
                        heapq.heappush(forward_pq, (cost + weight, neighbor, path + [node]))
        
        if backward_pq:
            cost, node, path = heapq.heappop(backward_pq)
            nodes_explored += 1
            
            if node not in backward_visited or cost < backward_visited[node][0]:
                backward_visited[node] = (cost, path + [node])
                
                if node in forward_visited:
                    total_cost = cost + forward_visited[node][0]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = forward_visited[node][1] + path[::-1][1:]
                
                for neighbor, weight in graph.get(node, {}).items():
                    if node_level[neighbor] >= node_level[node]:
                        heapq.heappush(backward_pq, (cost + weight, neighbor, path + [node]))
    
    return best_path, best_cost if best_path else None, nodes_explored


# Algorithm metadata
ALGORITHMS = {
    'bfs': {
        'name': 'Breadth-First Search (BFS)',
        'function': bfs,
        'pros': [
            'Simple implementation',
            'Guaranteed shortest path by hop count',
            'Complete algorithm',
            'Good for unweighted graphs'
        ],
        'cons': [
            'Ignores edge weights',
            'High memory usage',
            'Not optimal for weighted graphs',
            'Explores unnecessary nodes'
        ],
        'complexity': 'Time: O(V + E) | Space: O(V)',
        'best_for': 'Unweighted graphs, minimum hop count'
    },
    'dfs': {
        'name': 'Depth-First Search (DFS)',
        'function': dfs,
        'pros': [
            'Low memory usage',
            'Fast for tree structures',
            'Explores all paths',
            'Stack-based simplicity'
        ],
        'cons': [
            'No optimal path guarantee',
            'Can miss shorter paths',
            'Depends on graph structure',
            'May explore very deep first'
        ],
        'complexity': 'Time: O(V + E) | Space: O(V)',
        'best_for': 'Memory-constrained, path exploration'
    },
    'dijkstra': {
        'name': "Dijkstra's Algorithm",
        'function': dijkstra,
        'pros': [
            'Guaranteed optimal solution',
            'Industry standard',
            'Works for all non-negative weights',
            'Efficient with priority queue'
        ],
        'cons': [
            'Slower than A* (no heuristic)',
            'Cannot handle negative weights',
            'Explores all directions equally',
            'Higher computational cost'
        ],
        'complexity': 'Time: O((V + E) log V) | Space: O(V)',
        'best_for': 'GPS navigation, network routing'
    },
    'astar': {
        'name': 'A* (A-Star) Algorithm',
        'function': astar,
        'pros': [
            'Faster than Dijkstra',
            'Optimal with admissible heuristic',
            'Used in games and robotics',
            'Intelligent search direction'
        ],
        'cons': [
            'Requires heuristic function',
            'Memory intensive',
            'Heuristic quality matters',
            'Complex implementation'
        ],
        'complexity': 'Time: O((V + E) log V) | Space: O(V)',
        'best_for': 'Real-time navigation, gaming AI'
    },
    'bellman_ford': {
        'name': 'Bellman-Ford Algorithm',
        'function': bellman_ford,
        'pros': [
            'Handles negative weights',
            'Detects negative cycles',
            'Simple and reliable',
            'Distributed system friendly'
        ],
        'cons': [
            'Slow (O(VE))',
            'Not for real-time use',
            'High computational overhead',
            'Unnecessary for positive weights'
        ],
        'complexity': 'Time: O(V × E) | Space: O(V)',
        'best_for': 'Negative weights, cycle detection'
    },
    'floyd_warshall': {
        'name': 'Floyd-Warshall Algorithm',
        'function': floyd_warshall,
        'pros': [
            'All-pairs shortest paths',
            'Handles negative weights',
            'Simple dynamic programming',
            'Good for dense graphs'
        ],
        'cons': [
            'Very slow (O(V³))',
            'High memory (O(V²))',
            'Overkill for single queries',
            'Not scalable'
        ],
        'complexity': 'Time: O(V³) | Space: O(V²)',
        'best_for': 'Small graphs, all-pairs needed'
    },
    'johnson': {
        'name': "Johnson's Algorithm",
        'function': johnson,
        'pros': [
            'Efficient for sparse graphs',
            'Handles negative weights',
            'Better than Floyd-Warshall',
            'Clever reweighting'
        ],
        'cons': [
            'Complex implementation',
            'Overhead for single queries',
            'Preprocessing required',
            'Not needed if weights positive'
        ],
        'complexity': 'Time: O(V² log V + VE) | Space: O(V²)',
        'best_for': 'Sparse graphs with negative weights'
    },
    'mcsp': {
        'name': 'Multi-Criteria Shortest Path',
        'function': mcsp,
        'pros': [
            'Multiple factors (distance, accessibility)',
            'Realistic for real-world',
            'Customizable weights',
            'Better user experience'
        ],
        'cons': [
            'More complex',
            'Requires weight tuning',
            'Higher computational cost',
            'No unique optimal solution'
        ],
        'complexity': 'Time: O((V + E) log V) | Space: O(V)',
        'best_for': 'Accessibility routing, user preferences'
    },
    'dynamic': {
        'name': 'Dynamic Shortest Path',
        'function': dynamic_shortest_path,
        'pros': [
            'Real-time adaptation',
            'Handles obstacles',
            'Re-routing capability',
            'Practical for live systems'
        ],
        'cons': [
            'Requires monitoring',
            'Re-computation overhead',
            'Frequent updates needed',
            'Complex infrastructure'
        ],
        'complexity': 'Time: O((V + E) log V) per update',
        'best_for': 'Real-time navigation, emergencies'
    },
    'contraction_hierarchies': {
        'name': 'Contraction Hierarchies',
        'function': contraction_hierarchies,
        'pros': [
            'Extremely fast queries',
            'Production-ready (Google Maps)',
            'Scales to millions of nodes',
            'Bidirectional optimization'
        ],
        'cons': [
            'Requires preprocessing',
            'Not for dynamic graphs',
            'Complex implementation',
            'Preprocessing overhead'
        ],
        'complexity': 'Time: O(log V) query | Space: O(V)',
        'best_for': 'Large road networks, map services'
    }
}
