import geopandas as gpd
import numpy as np


# source: https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py
def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx


if __name__ == "__main__":
    # json_path = "D:/FSU OneDrive/OneDrive - Florida State University/datasets/nyc/NYC Taxi Zones.geojson"
    # save_path = "./nyc/adj.npy"
    #
    # gdf = geopandas.GeoDataFrame.from_file(json_path)
    # ctr = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
    # N = len(ctr)
    #
    # bike_area = [4, 12, 13, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 142, 143, 144, 148,
    #              158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249, 261]
    #
    # sensor_ids_ = bike_area  # list(range(N))
    # distance = []
    # for i in bike_area:
    #     for j in bike_area:
    #         distance.append([i, j, ctr[i - 1].distance(ctr[j - 1])])
    #
    # adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=sensor_ids_)
    # print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
    # np.save(save_path, adj_mx)

    """
    json_path = "/home/dy23a.fsu/jupyuter/safegraph/file/map_fl_2018.parquet"
    save_path = "/blue/gtyson.fsu/dy23a.fsu/datasets/panhandle/adj.npy"
    gdf =pd.read_parquet(json_path)
    gdf=gdf[gdf["COUNTYFP"].isin(["005","013","033","037","039","045","059","063","065","073","077","079","091","113","123","129","131","133"])]
    gdf = gpd.GeoDataFrame(gdf)
    gdf["geometry"]=gdf["geometry"].apply(lambda x: wkb.loads(x, hex=False))
    gdf=gdf.set_geometry("geometry").sort_values(by="GEOID")
    ctr = gdf.centroid.reset_index(drop=True)
    N = len(ctr)
    ph_area = list(range(N))
    distance = []
    for i in ph_area:
        for j in ph_area:
            distance.append([i, j, ctr[i].distance(ctr[j])])
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ph_area)
    print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
    np.save(save_path, adj_mx)
    """

    # json_path = "D:/OneDrive - Florida State University/datasets/nyc/taxi/NYC Taxi Zones.geojson"
    # save_path = "//data/manhattan/adj.npy"

    # gdf = geopandas.GeoDataFrame.from_file(json_path)
    # gdf=gdf[gdf["borough"]=="Manhattan"].drop_duplicates(subset="LocationID").reset_index(drop=True)
    # ctr = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
    # N = len(ctr)

    # bike_area = list(range(N))
    # sensor_ids_ = list(range(N))
    # distance = []
    # for i in bike_area:
    #     for j in bike_area:
    #         distance.append([i, j, ctr[i].distance(ctr[j])])

    # adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=sensor_ids_)
    # print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
    # np.save(save_path, adj_mx)

    # for db in ["fl","ny","ca","tx",]:
    #     gdf=read_map(db,2018)
    #     save_path = f"/blue/gtyson.fsu/dy23a.fsu/datasets/safegraph_{db}/adj.npy"
    #     gdf = gdf.set_geometry("geometry").sort_values(by="GEOID")
    #     ctr = gdf.centroid.reset_index(drop=True)
    #     N = len(ctr)
    #     ph_area = list(range(N))
    #     distance = []
    #     for i in ph_area:
    #         for j in ph_area:
    #             distance.append([i, j, ctr[i].distance(ctr[j])])
    #     adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ph_area)
    #     print(db, f"The shape of Adjacency Matrix: {adj_mx.shape}")
    #     np.save(save_path, adj_mx)

    gdf=gpd.GeoDataFrame.from_file("/blue/gtyson.fsu/dy23a.fsu/jupyter/mobility_datasets/nyc/Manhattan.geojson")
    save_path = f"/home/dy23a.fsu/st/datasets/nyc_mobility/Manhattan.npy"
    gdf = gdf.set_geometry("geometry")#.sort_values(by="GEOID")
    ctr = gdf.centroid.reset_index(drop=True)
    N = len(ctr)
    ph_area = list(range(N))
    distance = []
    for i in ph_area:
        for j in ph_area:
            distance.append([i, j, ctr[i].distance(ctr[j])])
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ph_area)
    print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
    np.save(save_path, adj_mx)

    gdf=gpd.GeoDataFrame.from_file("/home/dy23a.fsu/st/datasets/nyc_mobility_dense/Manhattan.geojson")
    save_path = f"/home/dy23a.fsu/st/datasets/nyc_mobility_dense/Manhattan.npy"
    gdf = gdf.set_geometry("geometry")#.sort_values(by="GEOID")
    ctr = gdf.centroid.reset_index(drop=True)
    N = len(ctr)
    ph_area = list(range(N))
    distance = []
    for i in ph_area:
        for j in ph_area:
            distance.append([i, j, ctr[i].distance(ctr[j])])
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ph_area)
    print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
    np.save(save_path, adj_mx)

    gdf=gpd.GeoDataFrame.from_file("/home/dy23a.fsu/st/datasets/chicago_mobility_dense/Chicago_dense.geojson")
    save_path = f"/home/dy23a.fsu/st/datasets/chicago_mobility_dense/chicago_adj_dense.npy"
    gdf = gdf.set_geometry("geometry")#.sort_values(by="GEOID")
    ctr = gdf.centroid.reset_index(drop=True)
    N = len(ctr)
    ph_area = list(range(N))
    distance = []
    for i in ph_area:
        for j in ph_area:
            distance.append([i, j, ctr[i].distance(ctr[j])])
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ph_area)
    print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
    np.save(save_path, adj_mx)