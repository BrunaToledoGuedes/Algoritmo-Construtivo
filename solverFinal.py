"""The solver for VRP(Vehicle Routing Problem).
The VRP(or mTSP) is NP-hard problem, therefore this algorithm uses heuristic approach as below:
author: louie
adaptado por: Bruna Toledo Guedes
"""
from operator import itemgetter, attrgetter
import sys
import math
import itertools
import time
from collections import namedtuple
import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

NUM_VEHICLES = 3

# Defines the data models.
Warehouse = namedtuple("Warehouse", ['index', 'x', 'y'])

Aterro = namedtuple("Aterro", ['index', 'x', 'y'])
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y', 'early_time', 'late_time', 'service_time', 'qtd'])
Vehicle = namedtuple("Vehicle", ['index', 'capacity', 'cost', 'x', 'y', 'customers', 'attributes'])


def read_csv_input_data(input_file_csv):
    """
    Reads csv input data file.
    :param input_file_csv:
    :return:
    """
    # Load the data
    locations_df = pd.read_csv(input_file_csv, delimiter=',', header=None, names=['latitude', 'longitude', 'early_time', 'late_time', 'service_time', 'qtd', 'is_customer'])
    # print(locations_df)

    is_warehouse = locations_df.is_customer == 0
    is_aterro = locations_df.is_customer == 2
    is_customer = locations_df.is_customer == 1
    subset_warehouse = locations_df[is_warehouse].dropna()
    subset_aterro = locations_df[is_aterro].dropna()
    subset_customer = locations_df[is_customer].dropna()
    print("warehouse: %s" % subset_warehouse)
    print("aterros: %s" % subset_aterro)
    print("# of customers: %s" % len(subset_customer))

    warehouses = []
    customers = []
    aterros = []
    warehouses.append(Warehouse(int(0), float(subset_warehouse.values[0][0]), float(subset_warehouse.values[0][1])))
    aterros.append(Aterro(int(0), float(subset_aterro.values[0][0]), float(subset_aterro.values[0][1])))
    aterros.append(Aterro(int(1), float(subset_aterro.values[1][0]), float(subset_aterro.values[1][1])))

    for i in range(0, len(subset_customer)):
        x = subset_customer.values[i][0]
        y = subset_customer.values[i][1]
        customers.append(Customer(int(i+1), int(1), float(x), float(y), subset_customer.values[i][2], subset_customer.values[i][3], subset_customer.values[i][4], subset_customer.values[i][5]))

    return warehouses, customers, aterros


def distance(point1, point2):
    """
    Calculates the Euclidean distance between two location coordinates.
    :param point1:
    :param point2:
    :return:
    """
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def tour_distance(tour, points):
    """
    Calculates the total tour distance between multiple coordinates.
    :param tour:
    :param points:
    :return:
    """
    return sum(distance(points[tour[i - 1]], points[tour[i]]) for i in range(len(tour)))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def plot_input_data(warehouses, customers, aterros):
    """
    Plots the input data.
    :param warehouses:
    :param customers:
    :return:
    """
    coords_warehouses = np.array([[c.x, c.y] for c in warehouses])
    coords_customers = np.array([[c.x, c.y] for c in customers])
    coords_aterros = np.array([[c.x, c.y] for c in aterros])

    plt.scatter(coords_customers[:, 0], coords_customers[:, 1], s=60, c='b', label='customer')
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', label='warehouse')
    plt.scatter(coords_aterros[:, 0], coords_aterros[:, 1], s=110, c='y', label='landfill')

    plt.legend()
    plt.grid()
    plt.show()


def plot_clusters(warehouses, customers, aterros, centroids, clusters, cluster_indexes_to_show):
    """
    Plots the clusters.
    :param warehouses:
    :param customers:
    :param centroids:
    :param clusters:
    :param cluster_indexes_to_show:
    :return:
    """

    #c=np.array([color])
    #c=color.reshape(1,-1)
    coords_warehouses = np.array([[c.x, c.y] for c in warehouses])
    coords_customers = np.array([[c.x, c.y] for c in customers])
    coords_aterros = np.array([[c.x, c.y] for c in aterros])
    
    #print("clusters")
    #print(clusters)
        
    #print("coords_customers")
    #print(coords_customers)
    
    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]
    
    #print("cluster_labels")
    #print(cluster_labels)
    
    cmap = plt.cm.get_cmap('Dark2')
    for i in range(len(cluster_labels)):
        if (i in cluster_indexes_to_show) or (cluster_indexes_to_show == []):
            color = cmap(1.0 * cluster_labels[i] / n_clusters)
            label_name = 'cluster' + str(i+1)
            # Plots the customers by each cluster.
            plt.scatter(coords_customers[clusters == i, 0], coords_customers[clusters == i, 1], s=60, c=color,
                        label=label_name)
            # Plots the centroid of each cluster.
            plt.scatter(centroids[i, 0], centroids[i, 1], s=240, c=color, marker='x', linewidths=1)

    # Plots the warehouse.
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', marker='s', label='warehouse')
    plt.scatter(coords_aterros[:, 0], coords_aterros[:, 1], s=110, c='y', label='landfill')

    plt.legend()
    plt.grid()
    plt.show()

    return


def plot_assigned_customers(warehouses, aterros, vehicles, vehicle_indexes_to_show):
    """
    Plots the assigned customers per vehicle.
    :param warehouses:
    :param vehicles:
    :param vehicle_indexes_to_show:
    :return:
    """
    coords_warehouses = np.array([[c.x, c.y] for c in warehouses])
    coords_aterros = np.array([[c.x, c.y] for c in aterros])

    cmap = plt.cm.get_cmap('Dark2')
    for i in range(0, len(vehicles)):
        vehicle = vehicles[i]
        if (i in vehicle_indexes_to_show) or (vehicle_indexes_to_show == []):
            color = cmap(1.0 * (i + 1) / len(vehicles))
            label_name = 'vehicle' + str(i+1)
            # Plots the allocated customers by each vehicle.
            coords_customers = np.array([[c.x, c.y] for c in vehicle.customers])
            # print('{0}: {1}'.format(label_name, coords_customers))
            print('{0}: {1} customers'.format(label_name, len(vehicle.customers)))
            plt.scatter(coords_customers[:, 0], coords_customers[:, 1], s=60, c=color,
                        label=label_name)
            # Plots the centroid of each cluster.
            plt.scatter(vehicle.x, vehicle.y, s=240, c='b', marker='x', linewidths=1)

    # Plots the warehouse.
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', marker='s', label='warehouse')
    plt.scatter(coords_aterros[:, 0], coords_aterros[:, 1], s=110, c='y', label='landfill')

    plt.legend()
    plt.grid()
    plt.show()

    return


def plot_vehicle_tour(vehicle, vehicle_tour):
    """
    Plots the vehicle's tour.
    :param vehicle:
    :param vehicle_tour:
    :return:
    """
    print("--------------> tour finale")
    print(vehicle_tour)
    # Plots the warehouse
    plt.scatter(vehicle_tour[0].x, vehicle_tour[0].y, s=240, c='r', marker='s', label='warehouse')
    # Plots the Aterro
    plt.scatter(vehicle_tour[1].x, vehicle_tour[1].y, s=240, c='r', marker='s', label='warehouse')
    plt.scatter(vehicle_tour[2].x, vehicle_tour[2].y, s=240, c='r', marker='s', label='warehouse')

    cmap = plt.cm.get_cmap('Dark2')
    prev_coords = vehicle_tour[0]
    arrow_head_width = 0.0004
    arrow_head_length = 0.0006
    for i in range(1, len(vehicle_tour) - 1):
        color = cmap(1.0 * (i + 1) / (len(vehicle_tour) - 2))
        customer = vehicle_tour[i]
        label_name = 'customer' + str(int(customer.index))
        # Plot the customer
        plt.scatter(customer.x, customer.y, s=240, c=color, label=label_name)

        dx = customer.x - prev_coords.x
        dy = customer.y - prev_coords.y

        if i == 1:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            xaxis_width = (xmax - xmin) / len(plt.xticks())
            yaxis_width = (ymax - ymin) / len(plt.yticks())
            arrow_head_width = yaxis_width / (len(plt.yticks()) * 20)
            arrow_head_length = xaxis_width / (len(plt.xticks()) * 20)
            arrow_width = 0.000001
            print('arrow_head_width:{0}, arrow_head_length:{0}'.format(str(arrow_head_width), str(arrow_head_length)))

        plt.arrow(prev_coords.x, prev_coords.y, dx, dy,
                  head_width=arrow_head_width, head_length=arrow_head_length, width=arrow_width,
                  fc='k', ec='k')
        prev_coords = customer

    dx_home = vehicle_tour[0].x - prev_coords.x
    dy_home = vehicle_tour[0].y - prev_coords.y
    plt.arrow(prev_coords.x, prev_coords.y, dx_home, dy_home,
              head_width=arrow_head_width, head_length=arrow_head_length, width=arrow_width,
              fc='k', ec='k')

    plt.title('Vehicle ' + str(vehicle.index + 1))
    plt.legend()
    plt.grid()
    plt.show()

    return


def detect_outliers(customers, percentile):
    """
    Detects the outliers.
    :param customers:
    :param percentile:
    :return:
    """
    # Find the global one centroid.
    clusters, centroids = cluster_customers(1, customers)
    centroid = Customer(0, 0, centroids[0][0], centroids[0][1], 0, 0, 0, 0)

    # Calculate the Euclidean distance between customer and centroid for all the customers.
    distances = []
    for customer in customers:
        dist = distance(centroid, customer)
        distances.append(dist)

    # Calculate the average distance.
    avg_distance = np.mean(distances)
    threshold_distance = np.percentile(distances, percentile)
    print('average distance from centroid = {0:.5f}'.format(avg_distance))
    print('threshold distance from centroid = {0:.5f}'.format(threshold_distance))

    # Detect the outliers if the Euclidean distance between customer and centroid is greater than average distance.
    inliers = []
    outliers = []
    for i in range(len(distances)):
        if distances[i] > threshold_distance:
            outliers.append(customers[i])
        else:
            inliers.append(customers[i])

    print('outliers: {0} of {1} ({2:.2f})'.format(len(outliers), len(customers), len(outliers)/float(len(customers))))
    return inliers, outliers


def cluster_customers(num_clusters, customers):
    """
    Clusters the customers.
    :param num_clusters:
    :param customers:
    :return:
    """
    kmeans = KMeans(n_clusters=num_clusters,
                    init='k-means++',   # 'random', 'k-means++'
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)

    coords = np.array([[c.x, c.y] for c in customers])
    y_km = kmeans.fit_predict(coords)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    centroids = kmeans.cluster_centers_
    print('clusters: %s' % cluster_labels)
    print('centroid: %s' % centroids)

    return y_km, centroids


def init_vehicles(warehouses, centroids, clusters, customers, max_capacity):
    """
    Initializes and sorts the cluster centroids(i.e. vehicles) by the nearest order of
    the distance between the warehouse and centroid.
    :param warehouses:
    :param centroids:
    :param clusters:
    :param customers:
    :param max_capacity:
    :return:
    """

    # Calculate the Euclidean distance between warehouse and each centroid.
    ordered_vehicles = []
    i = 0
    for centroid in centroids:
        # Get the customers in a cluster
        customers_in_cluster = []
        customers_array_in_cluster = np.array(customers)[clusters == i]
        for c in customers_array_in_cluster:
            customers_in_cluster.append(Customer(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]))

        dist = distance(warehouses[0], Customer(0, 0, centroid[0], centroid[1], 0, 0, 0, 0))
        vehicle = Vehicle(i, max_capacity, 0, centroid[0], centroid[1], customers_in_cluster, dist)
        ordered_vehicles.append(vehicle)
        i += 1
###Vehicle = namedtuple("Vehicle", ['index', 'capacity', 'cost', 'x', 'y', 'customers', 'attributes'])
    #print("veiculos")
    #print(vehicle)
    # Sort by distance ascending(i.e. from the nearest from the warehouse)
    # ordered_vehicles = sorted(ordered_vehicles, key=lambda x: x.attributes)
    # Sort by distance descending(i.e. from the farthest from the warehouse)
    ordered_vehicles = sorted(ordered_vehicles, key=lambda x: x.attributes, reverse=True)

    # print('ordered vehicles(centroids): %s' % ordered_vehicles)

    return ordered_vehicles


def assign_customers_to_vehicles(customers, vehicles, clusters, max_capacity):
    """
    Assigns the customers to vehicles.
    One customer will be allocated only into one vehicle.
    :param customers:
    :param vehicles:
    :param max_capacity:
    :return:
    """

    
    #print("assign_customers_to_vehicles")
    #print("clusters")
    #print(clusters)
    #print("customers")
    #print(customers)
    vehicles_ = []

    #shortage_capacity = len(customers) - len(vehicles) * max_capacity
    #if shortage_capacity > 0:
    #    # Allocate the additional shortage capacity to the first 40% vehicles.
    #    additional_capacity_vehicle = int(shortage_capacity / (len(vehicles) * 0.4))
    #    print('shortage capacity: {0}, additional capacity per vehicle: {1}'.format(shortage_capacity,
    #                                        additional_capacity_vehicle))

    i = 0
    for vehicle in vehicles:
        ordered_customers_tuple = []
        OrderedCustomer = namedtuple("ordered_customer", ['distance', 'data'])

        assigned_customers = []

        customers_in_cluster = vehicle.customers
        remaining_capacity = vehicle.capacity
        # if shortage_capacity > 0:
        #     remaining_capacity += additional_capacity_vehicle
        #     shortage_capacity -= additional_capacity_vehicle

        # Assign customers in the cluster first
        #print("customers_in_cluster")
        #print(customers_in_cluster)
        for customer_in_cluster in customers_in_cluster:
            if remaining_capacity-customer_in_cluster.qtd < 0:
                # vai para o aterro
                remaining_capacity = 280
                break
            #print("$$  remaining_capacity  $$")
            #print(remaining_capacity)
            for customer in customers:
                if customer.index == customer_in_cluster.index:
                    print('[assign(A)-vehicle{0}] remaining customers: {1}, remaining capacity: {2}'.format(int(i+1), len(customers), remaining_capacity))
                    assigned_customers.append(customer_in_cluster)
                    customers.remove(customer)
                    remaining_capacity -= customer_in_cluster.qtd
                    break

        # Calculate the Euclidean distance between customer and centroid of cluster(= centroid of vehicle)
        for customer in customers:
            dist = distance(customer, vehicle)
            ordered_customers_tuple.append(OrderedCustomer(dist, customer))

        # Sort by distance ascending(i.e. the nearest customers from vehicle)
        ordered_customers_tuple = sorted(ordered_customers_tuple, key=lambda x: x.distance)
        # Assign customers in the remaining by nearest distance order
        
        
        # agora eu acho que essa função ão serve pra nada...
        # somente durante o rotemento é que poderemos fazer as restrições...
        # vou abandonar essa função, por enquanto.
        
        
        # tá dando erro na linha abaixo
        # essa funcção tem que retornar a divisão certa dos centroides (veiculos)
        # considerando todas as possibilidades possiveis
        # capacidade, janela do cliente, hora de almoco do motorista 
        # horario de trabalho - tempo e deslocamento a 40km/hora
        # ver com mais calma as restrições.
        #
        # depois dessa função tem que rever a proxima, a gulosa.
        # e aí montar o roteiro considerando a necessidade de ir ao aterro  
        #  já sei que o terceiro veiculo tem que ir ao aterro... tem muitos clientes
        
        # analisar resultados - distancia percorrida, tempo de execução
        

       
  
        for j in range(0, remaining_capacity):
            customer = ordered_customers_tuple[j].data
            if j < len(ordered_customers_tuple):
                assigned_customers.append(customer)
                customers.remove(customer)
                remaining_capacity -= 1
                print('[assign(B)-vehicle{0}] remaining customers: {1}, remaining capacity: {2}'
                      .format(int(i + 1), len(customers), remaining_capacity))
                if len(customers) == 0:
                    break

        vehicle_ = Vehicle(i, len(assigned_customers), 0.0, vehicle.x, vehicle.y, assigned_customers, vehicle.attributes)
        print('* vehicle[{0}]: assigned {1} customers'.format(int(i+1), len(assigned_customers)))
        vehicles_.append(vehicle_)
        i += 1
        if len(customers) == 0:
            break

    # Assign the remaining customers to the nearest centroid(i.e. vehicle) if the vehicle capacity is available.
    print('number of unassigned customers = %d' % len(customers))
    if len(customers) > 0:
        unassigned_customers = []
        for customer in customers:
            nearest_vehicle = None
            min_distance = np.inf
            for vehicle in vehicles_:
                dist = distance(customer, vehicle)
                if dist <= min_distance and len(vehicle.customers) <= vehicle.capacity + additional_capacity_vehicle:
                    min_distance = dist
                    nearest_vehicle = vehicle
                    # print('min distance: %s' % str(min_distance))

            print('nearest vehicle: %s' % str(nearest_vehicle.index + 1))
            if nearest_vehicle is not None:
                nearest_vehicle.customers.append(customer)
                unassigned_customers.append(customer)

        for customer in unassigned_customers:
            customers.remove(customer)

    # Should be zero
    print('number of remaining customers = %d' % len(customers))
    # Check that the number of remaining customers is zero.
    assert len(customers) == 0
    return vehicles_


def greedy(points, veiculo):
    """
    Greedy optimization.
    :param points:
    :return:
    """
    point_count = len(points)
    #print(point_count)
    coords = np.array([(point.x, point.y) for point in points])
    coords_aterro = np.array([(points[1].x, points[1].y)])
    #capacity = np.array([(point.qtd) for point in points])
    tour = [0]
    candidates = set(range(2, point_count))
    hora_atual = 4.0 * 60 * 60 # 14400 hora de saida do deposito  
    hora_maxima_trabalho =  15.0 * 60 * 60  # 54000
    capacidade_restante = 280
    carga_carregada = 0
    janelaLate = 0
    janelaEarly = 0
    lunch = 0
    # velocidade do veiculo 40 mph = 17.88160 metros por segundo
    for neighborX in candidates:
        if points[neighborX].late_time < 2400:
           janelaLate = janelaLate + 1
        if points[neighborX].early_time > 0:
           janelaEarly = janelaEarly + 1
    # se janela > 1 então tem veiculos com restricao de horario
    distancia_percorrida = 0
    #print(tour)
    #sys.exit()
    while candidates:
        curr_point = tour[-1]
        nearest_neighbor = None
        nearest_dist = np.inf
        for neighbor in candidates:
            if ((coords[curr_point][0] - coords[neighbor][0])**2 + (coords[curr_point][1] - coords[neighbor][1])**2)**0.5 < nearest_dist:  # or primeiro == 1:
                if ((janelaLate + janelaEarly == 0 and ((points[neighbor].early_time)/100)*60*60 < hora_atual) or \
                   (janelaLate > 0 and points[neighbor].late_time < 2400.0 and \
                   ((points[neighbor].early_time)/100)*60*60 < hora_atual) or \
                   ((janelaEarly > 0 and points[neighbor].early_time == 0000.0 and \
                   ((points[neighbor].late_time)/100)*60*60 > hora_atual) and points[neighbor].late_time < 2400.0) or \
                   ((janelaEarly > 0 and points[neighbor].early_time > 0000.0 and \
                   ((points[neighbor].late_time)/100)*60*60 > hora_atual) and points[neighbor].late_time == 2400.0)):
                   if janelaLate + janelaEarly == 0:
                      nearest_neighbor = neighbor
                      nearest_dist = ((coords[curr_point][0] - coords[neighbor][0])**2 + (coords[curr_point][1] - coords[neighbor][1])**2)**0.5
                   else:
                      if (points[neighbor].early_time >= 0000.0 and points[neighbor].late_time < 2400.0) or \
                         (points[neighbor].early_time > 0000.0 and points[neighbor].late_time <= 2400.0):
                         nearest_neighbor = neighbor
                         nearest_dist = ((coords[curr_point][0] - coords[neighbor][0])**2 + (coords[curr_point][1] - coords[neighbor][1])**2)**0.5


            if nearest_neighbor != None:
               if points[nearest_neighbor].late_time < 0901.0 and ((points[nearest_neighbor].late_time)/100)*60*60 > hora_atual:
                  break  
               if points[nearest_neighbor].early_time >= 0800.0 and points[nearest_neighbor].late_time <= 1400.0 and ((points[nearest_neighbor].early_time)/100)*60*60 > hora_atual:
                  break  
        if nearest_neighbor == None:
          for neighbor in candidates:
            if ((coords[curr_point][0] - coords[neighbor][0])**2 + (coords[curr_point][1] - coords[neighbor][1])**2)**0.5 < nearest_dist:  # or primeiro == 1:
               if ((points[neighbor].early_time)/100)*60*60 < hora_atual and \
                   (points[neighbor].late_time/100)*60*60 > hora_atual: 
                       nearest_neighbor = neighbor
                       nearest_dist = ((coords[curr_point][0] - coords[neighbor][0])**2 + (coords[curr_point][1] - coords[neighbor][1])**2)**0.5

           
        # verificar hora de almoço
        # inicio entre 39600 e 43200
        if hora_atual > 39600 and lunch==0:
           lunch = 1
           print("Parada para almoço")
           print("Hora de inicio: ", hora_atual/3600)
           hora_atual = hora_atual + 3600
           print("Hora de término: ", hora_atual/3600)


        if nearest_neighbor == None:
           print("None Aguardando horário de atendimento dos clientes - hora atual: ", hora_atual/3600)
           hora_atual = hora_atual + 300
           #time.sleep(100)
           continue
        if (((points[nearest_neighbor].early_time)/100)*60*60 > hora_atual) or (((points[nearest_neighbor].late_time)/100)*60*60 < hora_atual):
            print("Aguardando horário de atendimento dos clientes - hora atual: ", hora_atual/3600)
            hora_atual = hora_atual + 300 # aguardando 5 minutos
            continue  #sys.exit()
        # verificar se o veículo ainda tem capacidade de coleta
        if capacidade_restante < points[nearest_neighbor].qtd:
            # se o carro estiver cheio, ir até o aterro e depois continuar
            tour.append(1)
            print("Veículo com capacidade esgotada. Alterando trajeto para descarregar no Aterro Sanitário!")
            print("Carga do veículo ", capacidade_restante)
            carga_carregada = carga_carregada + (280 - capacidade_restante)
            capacidade_restante = 280
            # somar 1600 segundos ao tempo por conta da descarga
            hora_atual = hora_atual + 1600
            # somar o tempo do transporte, até o aterro 
            distancia_percorrida = distancia_percorrida + ((coords[curr_point][0] - coords_aterro[0][0])**2 + (coords[curr_point][1] - coords_aterro[0][1])**2)**0.5 #(LA.norm(coords[curr_point] - coords_aterro[0]))
            print("hora atual antes ", hora_atual)
            hora_atual = hora_atual + ((((coords[curr_point][0] - coords_aterro[0][0])**2 + (coords[curr_point][1] - coords_aterro[0][1])**2)**0.5  / 17.88)) #((LA.norm(coords[curr_point] - coords_aterro[0]) / 17.88))
            print("hora atual depois ", hora_atual)
            # somar o tempo do transporte, do aterro ao proximo cliente
            distancia_percorrida = distancia_percorrida + ((coords_aterro[0][0] - coords[nearest_neighbor][0])**2 + (coords_aterro[0][1] - coords[nearest_neighbor][1])**2)**0.5 #(LA.norm(coords_aterro[0] - coords[nearest_neighbor]))
            print("hora atual antes ", hora_atual)
            #hora_atual = hora_atual + ((coords_aterro[0][0] - coords[nearest_neighbor][0])**2 + (coords_aterro[0][1] - coords[nearest_neighbor][1])**2)**0.5 #((LA.norm(coords_aterro[0] - coords[nearest_neighbor]) / 17.88))
            hora_atual = hora_atual + 1600 # service time para descarregar 
            print("hora atual depois ", hora_atual)
            
            # ATENÇÃO - desta não precisa somar o tempo de um cliente para o outro - ATENÇÃO
        else:
            # calcular tempo do trajeto 
            # veiculo começa o trabalho as 4 da manhã e tem que voltar as 15 horas.
            distancia_percorrida = distancia_percorrida + nearest_dist
            hora_atual = hora_atual + (nearest_dist / 17.88) # qtd metros por segundo - 40mph
            hora_atual = hora_atual + points[nearest_neighbor].service_time #service time do cliente
### alterei abaixo e comentei so pra testar
        #if not hora_atual > hora_maxima_trabalho:
        #   print("hora maxima de trabalho estourada")
        #   print("points")
        #   print(points)
        #   print("ponto corrente", curr_point)
        #   sys.exit()


        tour.append(nearest_neighbor)
        #print("nearest_neighbor")
        #print(points[nearest_neighbor].qtd)
        capacidade_restante = capacidade_restante - points[nearest_neighbor].qtd
        candidates.remove(nearest_neighbor)
        # verifica se ainda tem algum veiculo com restricao de janela
        janelaLate = 0
        janelaEarly = 0
        for neighborX in candidates:
            if points[neighborX].late_time < 2400:
               janelaLate = janelaLate + 1
            if points[neighborX].early_time > 0:
               janelaEarly = janelaEarly + 1
    print("------------------------------")
    print("Final do trajeto do veiculo ", veiculo)
    print("Carga final do veículo no dia em jardas ", carga_carregada + (280 - capacidade_restante))
    print("Tempo de trabalho efetivo no dia em segundos ", hora_atual-3600)
    print("Tempo de trabalho efetivo no dia em minutos ", (hora_atual-3600)/60)
    print("Tempo de trabalho efetivo no dia em horas ", (hora_atual-3600)/3600)
    print("===> Distancia percorrida ", distancia_percorrida)
    return tour_distance(tour, points), 0, tour


def swap(tour, dist, start, end, points):
    """
    Swap the points.
    :param tour:
    :param dist:
    :param start:
    :param end:
    :param points:
    :return:
    """
    #print("tour, dist, start, end, points")
    #print(end)
    new_tour = tour[:start] + tour[start:end + 1][::-1] + tour[end + 1:]

    new_distance = dist - \
                   (distance(points[tour[start - 1]], points[tour[start]]) +
                    distance(points[tour[end]], points[tour[(end + 1) % len(tour)]])) + \
                   (distance(points[new_tour[start - 1]], points[new_tour[start]]) +
                    distance(points[new_tour[end]], points[new_tour[(end + 1) % len(tour)]]))
    return new_tour, new_distance


def two_opt(points, veiculo):
    """
    2-opt optimization.
    :param points:
    :return:
    """
    point_count = len(points)-1
    best_distance, _, best_tour = greedy(points, veiculo)
    
    """
    improved = True
    t = time.process_time()
    # vai fazer todas as combinações possiveis de trajeto
    #   e encontrar o menor caminho, dentro 
    while improved:
        improved = False
        for start, end in itertools.combinations(range(point_count), 2):
            curr_tour, curr_distance = swap(best_tour, best_distance, start, end, points)
            if curr_distance < best_distance:
                best_tour = curr_tour
                best_distance = curr_distance
                improved = True
                break
        if time.process_time() - t >= 3600:   # 4 * 3600 + 59 * 60: 
            print("estourou o tempo *****************************************")
            improved = False
    """
    
    return tour_distance(best_tour, points), 0, best_tour


def plan_vehicle_routing(warehouse, aterro, vehicle):
    """
    Optimizes the vehicle routing.
    :param warehouse:
    :param vehicle:
    :return:
    """
    points = []
    points.append(warehouse)
    points.append(aterro)
    
    ### no programa original, ele praticamente descarta os clusters e considera todos
    ###   os veiculos no guloso.
    ### vou altera aqui e fazer com que ele considere somente os veiculos de cada cluster
    ###    e envia para o outro cluster somente se não foi possivel...
    a=sorted(vehicle.customers, key=itemgetter(5), reverse=False)
    #print("a", a)
    #sys.exit()
    for customer in a:
        points.append(customer)
    # Greedy solution (nearest neighbor)
    # Starts from 0, add nearest neighbor to the cycle at each step
    # Generally acceptable, but can be arbitrarily bad
    print("------------------------------")
    print("Inicio do trajeto do veiculo ", vehicle.index)
    best_distance, opt, best_tour = greedy(points, vehicle.index)
    #time.sleep(10)
    #print ("==========")
    #print("Veículo: ", vehicle.index)
    #print ("==========")
    # 2-opt solution
    #best_distance, opt, best_tour = two_opt(points, vehicle.index)
    #print('* best distance: {0}'.format(str(best_distance)))
    print('* best tour: {0}'.format(best_tour))

    # Calculate the cost of the solution
    cost = best_distance
    # cost += distance(warehouse, points[best_tour[0]])
    # cost += distance(warehouse, points[best_tour[len(best_tour) - 1]])
    #print('* total cost: {0}'.format(str(cost)))

    # Make directed cycle graph starting from the warehouse and returning to the warehouse.
    graph = []
    warehouse_index = len(best_tour)
    index = 0
    lefthand_vertices_of_warehouse = []
    for vertex in best_tour:
        if vertex == 0:  # Start from the warehouse
            warehouse_index = index
            graph.append(vertex)
        elif index > warehouse_index:
            graph.append(vertex)
        else:
            lefthand_vertices_of_warehouse.append(vertex)

        index += 1
    for vertex in lefthand_vertices_of_warehouse:
        graph.append(vertex)

    graph.append(0)  # Edge from the last vertex to the warehouse

    solution = []
    for vertex in graph:
        solution.append(points[vertex])

    return cost, opt, solution


def solve_vrp(warehouses, customers, aterros, is_plot):
    """
    Solves the vehicle routing problem.
    :param warehouses:
    :param customers:
    :param is_plot:
    :return:
    """
    # 1. EDA for input data.
    if is_plot is True:
        print("----------> Apresenta a configuração da instancia")
        plot_input_data(warehouses, customers, aterros)
    clusters, centroids = cluster_customers(NUM_VEHICLES, customers)
    if is_plot is True:
        print("----------> Apresenta a solução do kmeans")
        plot_clusters(warehouses, customers, aterros, centroids, clusters, [])

    
    # 2. Detect the outliers.
    # If the distance between global centroid and customer is outside of 85% percentile distance statistice,
    # set as outlier.
    ###inliers, outliers = detect_outliers(customers, 100)

    # 3. Find the centroids for 25 vehicles only with inliers.
    ###clusters, centroids = cluster_customers(NUM_VEHICLES, inliers)
    ###if is_plot is True:
    ###    plot_clusters(warehouses, inliers, aterros, centroids, clusters, [])

    # 4. Initialize and sort the cluster centroids by the farthest order of the distance
    # between the warehouse and centroid.
    # i.e. The sorted cluster centroids are the vehicles to assign the customers.
    # We assume that each vehicle's max capacity is 22 (i.e. capacity = number of customers / number of vehicles)
    max_capacity = 280 #len(customers) / NUM_VEHICLES
    print('max capacity = %d' % max_capacity)
    #vehicles = init_vehicles(warehouses, centroids, clusters, inliers, max_capacity)
    vehicles = init_vehicles(warehouses, centroids, clusters, customers, max_capacity)

    ### fazer o balanceamento do k-means
    ## veiculo 3 tem 21
    ## veiculo 1 tem 22
    ## veiculo 2 tem 56
    ##   total       99
    ### tem um veiculo a mais aí... talvez um deles seja um dos aterros, verificar.
    ## listar cada um dos veiculos e somar as cargas
        
    carga1 = 0
    carga2 = 0
    carga3 = 0
    ii1 = 0
    ii2 = 0
    ii3 = 0
    for vehicle in vehicles:
        #print(vehicle)
        if vehicle.index == 0:
           for customer in vehicle.customers:
               #print(vehicle.index, customer.index, customer.qtd)
               carga1 = carga1 + customer.qtd
               ii1 = ii1 + 1
        if vehicle.index == 1:
           for customer in vehicle.customers:
               #print(vehicle.index, customer.index, customer.qtd)
               carga2 = carga2 + customer.qtd
               ii2 = ii2 + 1
           # para o centroid 0
           for i in range(7):
              menorDistancia = 9999999999999
              indice = 0
              customer_escolhido = None
              #for customer in vehicle.customers:
              for iz, customer in enumerate(vehicle.customers):
                  distancia = ((centroids[0][0] - customer.x)**2 + (centroids[0][1] - customer.y)**2)**0.5
                  if menorDistancia > distancia:
                     menorDistancia = distancia
                     indice = iz
                     customer_escolhido = customer
              vehicle.customers.remove(customer_escolhido)

              for f in vehicles:
                  if f.index == 0:
                     f.customers.append(customer_escolhido)

           # para o centroid 2
           for i in range(7):
              menorDistancia = 9999999999999
              indice = 0
              customer_escolhido = None
              for iz, customer in enumerate(vehicle.customers):
                  distancia = ((centroids[2][0] - customer.x)**2 + (centroids[2][1] - customer.y)**2)**0.5
                  if i==1:
                     if customer.x == 1212329.0:
                          customer_escolhido = customer
                  else:
                     if menorDistancia > distancia:
                        menorDistancia = distancia
                        indice = iz
                        customer_escolhido = customer
              vehicle.customers.remove(customer_escolhido)

              for f in vehicles:
                  if f.index == 2:
                     f.customers.append(customer_escolhido)
                     
        if vehicle.index == 2:
           for customer in vehicle.customers:
               carga3 = carga3 + customer.qtd
               ii3 = ii3 + 1
    print("--------------------------------------------------0--1--2---")
    print("Antes do balanceamento - numero de clientes: ", ii1, ii2, ii3)
    print("Antes do balanceamento - carga de trabalho: ", carga1, carga2, carga3)

    # para finalizar pegar clientes com janela de tempo pequena para um veiculo com menos trabalho           


        
    carga1 = 0
    carga2 = 0
    carga3 = 0
    ii1 = 0
    ii2 = 0
    ii3 = 0
    for vehicle in vehicles:
        #print(vehicle)
        if vehicle.index == 0:
           for customer in vehicle.customers:
               #print(vehicle.index, customer.index, customer.qtd)
               carga1 = carga1 + customer.qtd
               ii1 = ii1 + 1               
        if vehicle.index == 1:
           for customer in vehicle.customers:
               #print(vehicle.index, customer.index, customer.qtd)
               carga2 = carga2 + customer.qtd
               ii2 = ii2 + 1
        if vehicle.index == 2:
           for customer in vehicle.customers:
               #print(vehicle.index, customer.index, customer.qtd)
               carga3 = carga3 + customer.qtd
               ii3 = ii3 + 1
    
    print("Depois do balanceamento - numero de clientes: ", ii1, ii2, ii3)
    print("Depois do balanceamento - carga de trabalho: ", carga1, carga2, carga3)
    print("--------------------------------------------------0--1--2---")

    # 6. Optimize the vehicle routing tour.
    output_data = ''
    total_cost = 0
    #print("vehicles")
    #print(vehicles)

    for vehicle in vehicles:
        # embora tenha  aterros, um fica muito longe, então só enviarei o primeiro, 
        #  que é o unico que ser´utilizado.
        obj, opt, vehicle_tour = plan_vehicle_routing(warehouses[0], aterros[0], vehicle)
        total_cost += obj
        # print("1 comecou a contar o total cost aqui...", total_cost)
        output_data += 'vehicle' + str(vehicle.index + 1) + ': ' + ' '.join([str(int(vertex.index)) for vertex in vehicle_tour]) + '\n'
        print("==========================================================================")
        print("-> Algoritmo greedy faz o trajeto pelo vizinho mais próximo,")
        print("   considerando as seguintes restrições:")
        print("        a) Capacidade máxima de cada um dos 3 veículos de 280 jardas")
        print("        b) Capacidade máxima diaria de um veículo de 400 jardas")
        print("        c) Total máximo de 500 paradas por dia, por veículo")
        print("        d) Hora de almoço de 1 hora começando entre 11 e 12 horas")
        print("        e) Velocidade de 40MPH")
        print("        f) Alguns clientes tem horários específicos - janela de visita")
        print("        g) O Depósito abre as 4 e fecha as 15 horas")
        print("==========================================================================")
        if is_plot is True:
            plot_vehicle_tour(vehicle, vehicle_tour)

    #output_data = 'total cost: %.5f' % total_cost + '\n' + output_data
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        input_file = sys.argv[1].strip()
        #print("----------linha 625")
        is_plot = str2bool(sys.argv[2].strip())

        warehouses, customers, aterros = read_csv_input_data(input_file)
        output = solve_vrp(warehouses, customers, aterros, is_plot)
        print(output)
    else:
        print('This requires an input file. (eg. python solver.py ../data/locations.csv true)')
