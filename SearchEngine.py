# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic to implement """
    # Heuristica 1: Distancia manhattan entre el nodo actual y el objetivo
    global NODES_EXPANDED
    h = 0
    # current_node y objective_node son tuplas (fila, columna) o (y, x)
    # Asegúrate de que se traten como tales para los cálculos
    y1, x1 = current_node
    y2, x2 = objective_node
    h = (abs(y1 - y2) + abs(x1 - x2)) * EPSILON # Multiplicar por EPSILON para asegurar admisibilidad y evitar costo cero en A* si los costos reales son pequeños
    NODES_EXPANDED += 1
    return np.float32(h)


def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic to implement is the distance of Euclide """
    global NODES_EXPANDED
    h = 0
    i, j = current_node
    i_k, j_k = objective_node
    distance = np.sqrt(pow((i-i_k), 2) + pow((j-j_k), 2)) * EPSILON
    h += distance
    NODES_EXPANDED += 1
    return h


def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors

    # Creamos el grafo dirigido
    G = nx.DiGraph()
    
    # Añadimos los nodos al grafo
    height_map, width_map = detection_map.shape

    # Añadimos los nodos al grafo con su probabilidad
    for i in range(height_map):
        for j in range(width_map):
            G.add_node((i, j), prob=float(detection_map[i, j]))

    # Añadimos las aristas al grafo
    for i in range(height_map):
        for j in range(width_map):
            current_node = (i, j) 
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)] # Arriba, Abajo, Izquierda, Derecha
            
            # Añadimos las aristas a los vecinos
            for ni, nj in neighbors:
                if 0 <= ni < height_map and 0 <= nj < width_map: 
                    neighbor_node = (ni, nj)
                    weight = G.nodes[neighbor_node]['prob']
                    # Solo agregar la arista si está dentro de la tolerancia
                    if weight <= tolerance:
                        G.add_edge(current_node, neighbor_node, weight=weight)
    return G




def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """Convierte coordenadas de (lat, lon) a índices (x, y) en la cuadrícula."""
    # Crear un array vacío para almacenar las coordenadas discretizadas
    discrete_coords = np.zeros((len(high_level_plan), 2), dtype=np.int32)
    # Validar la forma de entrada
    assert high_level_plan.shape[1] == 2, "high_level_plan debe tener forma (n, 2)"
    
    for i, coord in enumerate(high_level_plan):
        # Extraer latitud y longitud
        lat, lon = coord[0], coord[1]
        
        # Convertir a coordenada x (longitud), con redondeo para mejor distribución
        x = round((lon - boundaries.min_lon) / (boundaries.max_lon - boundaries.min_lon) * (map_width - 1))
        # Convertir a coordenada y (latitud), invertido para coincidir con el origen arriba a la izquierda
        y = round((boundaries.max_lat - lat) / (boundaries.max_lat - boundaries.min_lat) * (map_height - 1))
        
        # Asegurar que los índices estén dentro de los límites de la cuadrícula
        x = max(0, min(x, map_width - 1))
        y = max(0, min(y, map_height - 1))
        
        # Almacenar las coordenadas discretizadas
        discrete_coords[i] = [y, x]
    
    return discrete_coords



def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32,
                 initial_location_index: int = 0) -> tuple:
    """Busca un camino que pase por todos los POIs en orden usando A* para cada tramo."""
    global NODES_EXPANDED
    NODES_EXPANDED = 0

    # Discretizar los POIs
    discretized_locations = discretize_coords(
        high_level_plan=locations,
        boundaries=boundaries,
        map_width=map_width,
        map_height=map_height
    )

    # Inicializar el camino completo
    full_path = []
    
    # Si solo hay un POI, devolverlo como solución
    if len(discretized_locations) == 1:
        print("Solo hay un POI disponible para visitar.")
        return [[tuple(discretized_locations[0])]], NODES_EXPANDED
    
    # Si hay más de un POI, buscar el camino entre ellos
    for idx in range(len(discretized_locations) - 1):
        start = tuple(discretized_locations[idx])  # Nodo inicial
        end = tuple(discretized_locations[idx + 1])  # Nodo final
        
        try:
            # Buscar el camino entre los nodos usando A*
            partial_path = nx.astar_path(
                G,
                start,
                end,
                heuristic=lambda n1, n2: heuristic_function(n1, n2),
                weight='weight'
            )
            # Evita repetir el nodo de unión
            if full_path and partial_path[0] == full_path[-1]:
                partial_path = partial_path[1:]

            # Añadir el camino parcial al camino completo
            full_path.extend(partial_path)
    
        except nx.NetworkXNoPath:
            # Si no hay camino entre dos POIs, imprime un mensaje y devuelve una solución vacía
            print(f"No existe un camino disponible entre los puntos {start} y {end} con la tolerancia actual. "
                  f"Prueba aumentando el parámetro de tolerancia.")
            return [], NODES_EXPANDED

    # Adaptar el formato: lista de listas de strings (cada string es una tupla)
    if full_path:
        solution_plan_str = [[str(node) for node in full_path]]
    else:
        solution_plan_str = [[]]
    return solution_plan_str, NODES_EXPANDED


def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Calcula el costo total de la solución de planificación completa """
    costo_total = 0.0

    # Si solution_plan es una lista de listas de strings, conviértelo a lista de tuplas
    if solution_plan and isinstance(solution_plan[0], list):
        # Solo tomamos la primera ruta (asumiendo un solo camino)
        nodos = [eval(nodo_str) for nodo_str in solution_plan[0]]
    else:
        nodos = solution_plan

    # Iterar sobre los nodos en el plan de solución y sumar los costos
    for i in range(len(nodos) - 1):
        origen = nodos[i] # Nodo de origen
        destino = nodos[i + 1] # Nodo de destino
        if G.has_edge(origen, destino):  # Verificar si el borde existe
            costo_total += G[origen][destino]['weight']  # Sumar el costo del borde
        else:  
            print(f"Borde faltante entre {origen} y {destino} en el plan de solución.")

    return np.float32(costo_total)