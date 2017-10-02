import numpy as np
import networkx as nx
from collections import Counter
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import pairwise_distances


def comm_ave_pairwise_spatial_dist(partitions, locations):
    """
    Community Average Pairwise Spatial Distance

    Source: comm_ave_pairwise_spatial_dist.m
    Reference: Feldt Muldoon S, Soltesz I, Cossart R (2013) Spatially clustered
        neuronal assemblies comprise the microstructure of synchrony in chronically
        epileptic networks. Proc Natl Acad Sci USA 110:3567?3572.

    Inputs
    ======
    partitions: list or array of len N, N is number of nodes in network
        Each entry contains the community index for
        node i (where communities are sequentially indexed as 1:M and M
        is the total number of detected communities).
    locations: list or array of size N x dims,

    Outputs
    =======
    comm_ave_pairwise_spatial_dist: (M+1) x 2 np.array where M is the number
        of communities. The first column contains the community index and
        the second column contains the average pairwise spatial distance
        within that community.  The M+1th entry contains the average pairwise
        spatial distance of the network (denoted as community 0).

    Example
    =======
    >> partitions = [1, 2, 1, 1, 2, 3, 3]
    >> locations = np.array([[0, 1], [3,0], [0, 0],
                             [0, 2], [5, 0], [-1, -1],
                             [-1, -2]])
    >> comm_ave_pairwise_spatial_dist(partitions, locations)

    """
    partitions = np.array(partitions).ravel()
    locations = np.array(locations)

    comm_ave_pairwise_spatial_dist = []
    d = pairwise_distances(locations, metric='euclidean')
    comm_avg = d[np.triu_indices(len(d), k=1)].mean() # mean of lower diag
    # alternatively
    # np.fill_diagonal(d, np.nan)
    # comm_avg = np.nanmean(d)
    comm_ave_pairwise_spatial_dist.append((0, comm_avg))
    for partition in np.unique(partitions):
        if len(np.where(partitions == partition)[0]):
            d = pairwise_distances(locations[np.where(partitions == partition)], metric='euclidean')
            comm_avg = d[np.triu_indices(len(d), k=1)].mean()
            comm_ave_pairwise_spatial_dist.append((partition, comm_avg))
        else:
            comm_ave_pairwise_spatial_dist.append((partition, 0))
    return np.array(comm_ave_pairwise_spatial_dist)


def comm_laterality(partitions, categories):
    """
    Function to calculate the laterality of detected communities

    Reference: Doron KW, Bassett DS, Gazzaniga MS (2012) Dynamic network
        structure of interhemispheric coordination. Proc Natl Acad Sci USA
        109:18661?18668.

    Inputs
    ======
    partitions: an Nx1 array where N is the number of nodes
        in the network.  Each entry contains the communitiy index for
        node i (where communities are sequentially indexed as 1:M and M
        is the total number of detected communities).

    categories: a Nx1 array where N is the number of nodes and
        each entry is either a '0' or '1' denoting the assignment of each
        node to one of two communities.

    Outputs
    =======
    comm_laterality_array: an (M+1)x2 array where M is the number
        of communities.  The first column contains the community index and
        the second column contains the laterality of that community.
        The M+1th entry contains the  laterality of the partition of the
        network (denoted as community 0).
    """

    partitions = np.array(partitions).ravel()
    categories = np.array(categories).ravel()
    categories_list = list(set(categories))

    n_nodes = len(partitions)
    n_communities = len(np.unique(partitions))
    n_surrogates = 1000

    comm_laterality = []
    number_nodes_in_communities = []
    for partition in np.unique(partitions):
        categories_i = categories[np.where(partitions == partition)]
        n_nodes_in_i = len(categories_i)
        categories_count = Counter(categories_i)
        n_0 = categories_count.get(categories_list[0], 0)
        n_1 = categories_count.get(categories_list[1], 0)
        laterality = np.abs(n_0 - n_1)/n_nodes_in_i
        number_nodes_in_communities.append((n_nodes_in_i, n_0, n_1))
        comm_laterality.append((partition, laterality))
    number_nodes_in_communities = np.array(number_nodes_in_communities)
    total_number_in_comm_1 = np.sum(number_nodes_in_communities[2, :])

    surrogate_laterality = []
    for _ in range(n_surrogates):
        randomized_categories = np.zeros(shape=(n_nodes), dtype=np.int)
        rand_perm_assignment = np.random.permutation(range(n_nodes))
        place_in_comm_1 = rand_perm_assignment[0:total_number_in_comm_1]
        randomized_categories[place_in_comm_1] =  1

        for partition in np.unique(partitions):
            categories_i = categories[np.where(partitions == partition)]
            n_nodes_in_i = len(categories_i)
            n_0 = categories_count.get(categories_list[0], 0)
            n_1 = categories_count.get(categories_list[1], 0)
            rand_laterality  = np.abs(n_0 - n_1) / n_nodes_in_i
            surrogate_laterality.append((partition, rand_laterality))

    surrogate_laterality = np.array(surrogate_laterality)
    expected_comm_laterality = np.sum(surrogate_laterality,axis = 1) / n_surrogates

    net_laterality = (1/n_nodes) * np.dot(number_nodes_in_communities[:, 0], comm_laterality[0:n_communities, 1]) - expected_comm_laterality
    comm_laterality.append((0, net_laterality))

    return np.array(comm_laterality_array)


def comm_radius(partitions, locations):
    """
    Function to calculate the radius of detected communities

    Reference:  Lohse C, Bassett DS, Lim KO, Carlson JM (2014) Resolving
        anatomical and functional structure in human brain organization:
        identifying mesoscale organization in weighted network representations.
        PLoS Comput Biol 10:e1003712.

    Inputs
    ======
    partitions: an Nx1 array where N is the number of nodes
        in the network.  Each entry contains the community index for
        node i (where communities are sequentially indexed as 1:M and M
        is the total number of detected communities).

    locations: a Nxdim array where N is the number of nodes and
        dim is the spatial dimensionality of the network (1,2,or 3).  The
        columns contain the (x,y,z) coordinates of the nodes in euclidean
        space

    Outputs
    =======
    comm_radius_array: an (M+1)x2 array where M is the number
        of communities.  The first column contains the community index and
        the second column contains radius of that community.
        The 0 th entry contains the average community radius of the
        network (denoted as community 0).
    """
    comm_radius_array = []

    # diameter of entire network
    comm_radius = pairwise_distances(locations).max()
    comm_radius_array.append((0, comm_radius))

    # diameter for each community
    for partition in np.unique(partitions):
        nodes_in_i = np.where(partitions == partition)
        n_nodes_in_i =  len(np.where(partitions == partition)[0])
        if n_nodes_in_i >= 2:
            comm_nodes = locations[nodes_in_i]
            comm_radius = pairwise_distances(comm_nodes).max()
            comm_radius_array.append((partition, comm_radius))
        else:
            comm_radius_array.append((partition, 0))

    return comm_radius_array


def comm_spatial_diameter(partitions,locations):
    """
    Function to calculate the spatial diameter of detected communities.

    Reference: Feldt Muldoon S, Soltesz I, Cossart R (2013) Spatially clustered
        neuronal assemblies comprise the microstructure of synchrony in chronically
        epileptic networks. Proc Natl Acad Sci USA 110:3567?3572.

    Inputs
    ======
    partitions: an Nx1 array where N is the number of nodes
        in the network.  Each entry contains the community index for
        node i (where communities are sequentially indexed as 1:M and M
        is the total number of detected communities).

    locations: a Nxdim array where N is the number of nodes and
        dim is the spatial dimensionality of the network (1,2,or 3).  The
        columns contain the (x,y,z) coordinates of the nodes in euclidean
        space

    Outputs
    =======
    comm_spatial_diameter_array: an (M+1) x 2 array where M is the number
        of communities.  The first column contains the community index and
        the second column contains the spatial diameter
        within that community.  The 0th entry contains the
        spatial diameter of the network (denoted as community 0).

    """
    comm_spatial_diameter_array = []

    n_nodes = len(partitions)
    n_communities = len(np.unique(partitions))
    communities = np.unique(partitions)
    locations = np.array(locations)

    # calculate entire network
    max_dist = pairwise_distances(locations).max()
    comm_spatial_diameter_array.append((0, max_dist)) # community 0th

    # calculate each community
    for partition in communities:
        nodes_in_i = np.where(partitions == partition)[0]
        dist = []
        if len(nodes_in_i) > 1:
            for community in communities:
                node_in_j = np.where(partitions == community)[0]
                dist.append(list(pairwise_distances(nodes_in_i, node_in_j).ravel()))
                max_dist_i = max(dist)
                comm_spatial_diameter_array.append((partition, max_dist_i))
    return np.array(comm_spatial_diameter_array)


def comm_spatial_extent(partitions, locations):
    """
    Function to calculate the spatial extent of detected communities

    Inputs
    ======
    partitions: an N x 1 array where N is the number of nodes
        in the network.  Each entry contains the community index for
        node i (where communities are sequentially indexed as 1:M and M
        is the total number of detected communities).

    locations: a N x dim array where N is the number of nodes and
        dim is the spatial dimensionality of the network (2,or 3).  The
        columns contain the (x,y,z) coordinates of the nodes in euclidean
        space

    Outputs
    =======
    comm_spatial_extent_array: an (M+1) x 2 array where M is the number
        of communities.  The first column contains the community index and
        the second column contains the spatial extent
        of that community.  The M+1th entry contains the
        spatial extent of the network (denoted as community 0).
    """

    partitions = np.array(partitions).ravel()
    communities = np.unique(partitions)
    n_nodes, n_dim = np.array(locations).shape
    comm_spatial_extent_array = []

    # spatial extent for entire network
    cvh = ConvexHull(locations)
    spatial_extent_network = cvh.volume / n_nodes
    comm_spatial_extent_array.append((0, spatial_extent_network))

    # spatial extent for each community
    for partition in communities:
        nodes_in_i = locations[np.where(partitions == partition)]
        n_nodes_in_i = len(nodes_in_i)
        if n_nodes_in_i < n_dim + 1:
            volume = 0
        else:
            cvh = ConvexHull(nodes_in_i)
            volume = cvh.volume
        comm_spatial_extent_array.append((partition, volume / n_nodes_in_i))
    return np.array(comm_spatial_extent_array)


def q_value(C, A):
    """
    This is a function that calculates modularity
    """
    if len(C) == 1:
        C = C.T
    n_nodes = len(A)
    cl_label = np.unique(C)
    num_cl = len(cl_label)
    cl = np.zeros((num_node, num_cl))
    for i in range(num_cl):
        cl[:, i] = (C == cl_label[i]).astype(float);
    Q_mat = ((cl.T).dot(A)).dot(cl)
    return Q_mat


def sig_perm_test(C, A, T=10):
    """
    Uses the permutation test to calculate
    the significance of clusters in a given community structure.

    Inputs
    ======
    A: a N-by-N weighted adjacency matrix
    C: a N-by-1 partition(cluster) vector
    T: # of random permutations

    Outputs
    =======
    sig_array: the significance of all clusters
    Q: the modularity of the given partition(cluster)
    Q_r: the modularities of all random partitions
    """

    n_nodes = len(A)
    n_communities = len(np.unique(C))
    Q = q_value(C, A)
    Q_r = np.zeros((T, n_communities, n_communities), dtype=np.float)
    for i in range(T):
        C_r = C[np.random.permutation(range(n_nodes))]
        Q_r[i,:,:] = q_value(C_r, A)
    Q_avg = np.zeros((n_communities, n_communities), dtype=np.float)
    Q_std = np.zeros((n_communities, n_communities), dtype=np.float)
    for i in range(num_cl):
        for j in range(num_cl):
            temp = Q_r[:, i, j].flatten()
            Q_avg[i,j] = np.mean(temp)
            Q_std[i,j] = np.std(temp)
    sig_array = np.divide((Q_mat - Q_avg), Q_std)

    return (sig_array, Q, Q_r)


def sig_lmc(C, A):
    """
    This a function that uses the Lumped Markov chain to calculate
    the significance of clusters in a given community structure.
    refer to "Piccardi 2011 in PloS one".

    Here we normalize the original definition of persistence by
    the size of the corresponding cluster

    Inputs
    ======
    A: a N-by-N weighted adjacency array
    C: a N-by-1 partition(cluster) vector

    Outputs
    =======
    persistence: normalized persistence probability of all clusters

    """
    C = C.ravel()
    n_nodes = len(A)
    n_communities = len(np.unique(C))
    communities = np.unique(C)

    P = np.linalg.solve(np.diag(A.sum(axis=1)), A)
    eig_val, eig_vec = np.linalg.eig(P.T)
    if eig_vec.min() < 0:
        eig_vec = -eig_vec
    Pi = eig_vec.T
    H = np.zeros((n_nodes, n_communities))
    for c in communities:
        H[:, i] = (C == c).astype(float)

    R = np.diag(P.dot(H).ravel())
    Q = R.dot(np.diag(P)).dot(P).dot(H)
    persistence = np.divide(np.multiply(diag(Q), H.sum(axis=0)), H.T.sum(axis=0))

    return persistence


def flexibility(S, nettype='temp'):
    """
    Flexibility coefficient

    calculates the flexibility coefficient of S.
    The flexibility of each node corresponds to the number of times that
    it changes module allegiance, normalized by the total possible number
    of changes. In temporal networks, we consider changes possible only
    between adjacent time points. In multislice/categorical networks,
    module allegiance changes are possible between any pairs of slices.

    Inputs
    ======
    S: pxn matrix of community assignments where p is the
        number of slices/layers and n the number of nodes
    nettype: string specifying the type of the network:
        'temp'  temporal network (default)
        'cat'   categorical network

    Outputs
    =======
    F: Flexibility coefficient

    """

    n_slices, n_nodes = S.shape
    if nettype == 'temp':
        possible_changes = n_slices - 1
        for i in range(1, len(S)):
            total_changes += np.sum(S[i, :] == S[i - 1, :])
    elif nettype == 'cat':
        possible_changes = n_slices * (n_slices - 1)
        for s in range(len(S)):
            other_slices = [i for i in range(10) if i != s]
            total_changes = np.sum(np.tile(S[s, :], (n_slices - 1, 1)) != S[other_slices, :])
    else:
        return None

    F = total_changes / possible_changes

    return F


def promiscuity(S):
    """
    Promiscuity coefficient

    Calculates the promiscuity coefficient. The
    promiscuity of a temporal or multislice network corresponds to the
    fraction of all the communities in the network in which a node
    participates at least once.

    Inputs
    ======
    S: (p x n) array of community assignments where p is the
        number of slices/layers and n the number of nodes

    Outputs
    =======
    P: Promiscuity coefficient
    """

    P = []
    n_slices, n_nodes = S.shape
    for n in range(n_nodes):
        p = (len(np.unique(S[:, n])) - 1) / (n_slices - 1)
        P.append(p)

    return np.array(P)


def persistence(S):
    """
    This function computes the persistence for a given multilayer partition S
    defined in  "Community detection in temporal multilayer networks, and ...
    its application to correlation networks" (http://arxiv.org/abs/1501.00040)

    Inputs
    ======
    S: pxn matrix, n is the numebr of partitions, p is the number of nodes

    Outputs
    =======
    pers: the persistence value
    """

    pers = (S[:, 1::] == S[:, 0:-1]).sum()

    return pers


def integration(MA, system_by_node):
    """
    Integration coefficient calculates the integration coefficient
    for each node of the network. The integration coefficient of a region
    corresponds to the average probability that this region is in the same
    network community as regions from other systems.

    Inputs
    ======
    MA, Module Allegiance matrix, where element (i,j)
        represents the probability that nodes i and j
        belong to the same community
    system_by_node, np.array array containing the system
        assignment for each node

    Outputs
    =======
    I, integration coefficient for each node (M x 2 array)

    Marcelo G Mattar (08/21/2014)
    """

    I = []
    MA = np.array(MA)
    system_by_node = np.array(system_by_node).ravel()
    np.fill_diagonal(MA, np.nan)

    for s in np.unique(system_by_node):
        idx = np.where(system_by_node != s)[0]
        int_coef = np.nanmean(MA[idx[:, None], idx])
        I.append((s, int_coef))

    return np.array(I)


def consensus_iterative(C):
    """
    Construct a consensus (representative) partition using the iterative thresholding procedure

    Identifies a single representative partition from a set of C partitions,
    based on statistical testing in comparison to a null model.
    A thresholded nodal association matrix is obtained by subtracting a random nodal
    association matrix (null model) from the original matrix. The
    representative partition is then obtained by using a Generalized
    Louvain algorithm with the thresholded nodal association matrix.

    Inputs
    ======
    C: pxn matrix of community assignments where p is the
        number of optimizations and n the number of nodes

    Outputs
    =======
    S2: pxn matrix of new community assignments
    Q2: associated modularity value
    X_new3: thresholded nodal association matrix
    qpc: quality of the consensus (lower == better)

    Reference:
    http://commdetect.weebly.com/uploads/4/9/5/9/49593677/consensus_iterative.m
    """
    return


def zrand(part1, part2):
    """
    Calculates the z-Rand score and Variation of Information
    distance between a pair of partitions.

    Calculates the z-score of the
    Rand similarity coefficient between partitions part1 and part2. The
    Rand similarity coefficient is an index of the similarity between the
    partitions, corresponding to the fraction of node pairs identified the
    same way by both partitions (either together in both or separate in
    both)

    Inputs
    ======
    part1, Partitions that are being
    part2, compared with one another

    Outputs
    =======
    zRand, z-score of the Rand similarity coefficient
    SR, Rand similarity coefficient
    SAR, Adjusted Rand similarity coefficient
    VI, Variation of information

    Reference:
    Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter,
    "Comparing Community Structure to Characteristics in Online Collegiate
    Social Networks," SIAM Review 53, 526-543 (2011).

    https://github.com/nangongwubu/Python-Version-for-Network-Community-Architecture-Toobox/blob/master/ncat.py#L14-L128
    """

    part1 = np.array(part1).ravel()
    part2 = np.array(part2).ravel()
    n = len(part1)
    if part1.shape != part2.shape:
        print('ERROR: partitions not of equal length')
        return
    n_1 = len(set(part1)) + 1
    n_2 = len(set(part2)) + 1
    nij = sparse.csr_matrix((np.ones(part1.shape, dtype=int), (part1, part2)),
                             shape=(n_1, n_2))
    ni = nij.sum(axis=1)
    nj = nij.sum(axis=0)
    nj = nj.T

    M = n * (n - 1)/2.
    M1 = ((ni.multiply(ni) - ni) / 2.).sum()
    M2 = ((nj.multiply(nj) - nj) / 2.).sum()


    a = ((nij.multiply(nij) - nij) / 2.).sum()
    b = M1 - a
    c = M2 - a
    d = M - (a + b + c)
    SR=(a + d) / (a + b + c + d)
    meana = M1 * M2 / M
    SAR = (a - meana) / ((M1 + M2)/2. - meana)

    C1 = 4*((np.power(ni, 3)).sum()) - 8 * (n+1) * M1 + n * (n * n - 3 * n - 2)
    C2 = 4*((np.power(nj, 3)).sum()) - 8 * (n+1) * M2 + n * (n * n - 3 * n - 2)

    vara = M/16 - np.power((4 * M1 - 2 * M), 2) * np.power((4 * M2 - 2 * M), 2)/(256 * M * M) + \
        C1 * C2/(16 * n * (n - 1) * (n - 2)) + \
        (np.power((4 * M1 - 2 * M), 2) - 4 * C1 - 4 * M) * \
        (np.power((4 * M2 - 2 * M),2) - 4 * C2 - 4 * M)/(64 * n * (n-1) * (n-2) * (n-3))

    zRand = (a - meana) / np.sqrt(vara)

    c1 = set(part1);
    c2 = set(part2);
    H1, H2, I = 0, 0, 0
    for i in c1:
        pi = np.double(ni[i])/n
        H1 = H1 - pi * np.log(pi)
        for j in c2:
            if nij[i,j]:
                pj = np.double(nj[j])/n;
                pij = np.double(nij[i,j])/n;
                I = I + pij * np.log(pij/ (pi * pj))
    for j in c2:
        pj = np.double(nj[j]) / n
        H2 = H2 - pj * np.log(pj)
    VI = (H1 + H2 - 2 * I)
    return zRand, SR, SAR, VI


def consensus_similarity(C):
    """
    Construct a consensus (representative) partition
    using the iterative thresholding procedure

    identifies a single representative partition from a set of C partitions
    that is the most similar to the all others. Here, similarity is taken
    to be the z-score of the Rand coefficient (see zrand.m)


    Inputs
    ======
    C, pxn matrix of community assignments where p is the
        number of optimizations and n the number of nodes

    Outputs
    =======
    consensus, consensus partition
    consensus_simm,	average similarity between consensus
        partition and all others
    pairwise_simm,	pairwise similarity matrix
    """

    n_part, _ = C.shape
    pairwise_simm = np.zeros((n_part, n_part))
    for i in range(n_part):
        for j in range(n_part):
            pairwise_simm[i, j] = zrand(C[i, :], C[j, :])
    pairwise_simm = pairwise_simm + pairwise_simm.T

    average_pairwise_simm = np.sum(pairwise_simm, axis=1) / (n_part - 1)
    idx = np.argmax(average_pairwise_simm)
    consensus = C[idx, :]
    consensus_simm = max(average_pairwise_simm)

    return consensus, consensus_simm, pairwise_simm


def community_louvain(W, gamma=1, ci=None, B='modularity', seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This function is a fast an accurate multi-iterative generalization of the
    louvain community detection algorithm. This function subsumes and improves
    upon modularity_[louvain,finetune]_[und,dir]() and additionally allows to
    optimize other objective functions (includes built-in Potts Model i
    Hamiltonian, allows for custom objective-function matrices).
    Parameters
    ----------
    W : NxN np.array
        directed/undirected weighted/binary adjacency matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
        ignored if an objective function matrix is specified.
    ci : Nx1 np.arraylike
        initial community affiliation vector. default value=None
    B : str | NxN np.arraylike
        string describing objective function type, or provides a custom
        NxN objective-function matrix. builtin values
            'modularity' uses Q-metric as objective function
            'potts' uses Potts model Hamiltonian.
            'negative_sym' symmetric treatment of negative weights
            'negative_asym' asymmetric treatment of negative weights
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.array
        final community structure
    q : float
        optimized q-statistic (modularity only)

    Reference:
    https://github.com/aestrivex/bctpy/blob/master/bct/algorithms/modularity.py#L66-L233
    '''
    np.random.seed(seed)

    n = len(W)
    s = np.sum(W)

    if np.min(W) < -1e-10:
        raise BCTParamError('adjmat must not contain negative weights')

    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            raise BCTParamError('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()

    if B in ('negative_sym', 'negative_asym'):
        W0 = W * (W > 0)
        s0 = np.sum(W0)
        B0 = W0 - gamma * np.outer(np.sum(W0, axis=1), np.sum(W, axis=0)) / s0

        W1 = W * (W < 0)
        s1 = np.sum(W1)
        if s1:
            B1 = (W1 - gamma * np.outer(np.sum(W1, axis=1), np.sum(W1, axis=0))
                / s1)
        else:
            B1 = 0

    elif np.min(W) < -1e-10:
        raise BCTParamError("Input connection matrix contains negative "
            'weights but objective function dealing with negative weights '
            'was not selected')

    if B == 'potts' and np.any(np.logical_not(np.logical_or(W == 0, W == 1))):
        raise BCTParamError('Potts hamiltonian requires binary input matrix')

    if B == 'modularity':
        B = W - gamma * np.outer(np.sum(W, axis=1), np.sum(W, axis=0)) / s
    elif B == 'potts':
        B = W - gamma * np.logical_not(W)
    elif B == 'negative_sym':
        B = B0 / (s0 + s1) - B1 / (s0 + s1)
    elif B == 'negative_asym':
        B = B0 / s0 - B1 / (s0 + s1)
    else:
        try:
            B = np.array(B)
        except:
            raise BCTParamError('unknown objective function type')

        if B.shape != W.shape:
            raise BCTParamError('objective function matrix does not match '
                                'size of adjacency matrix')
        if not np.allclose(B, B.T):
            print ('Warning: objective function matrix not symmetric, '
                   'symmetrizing')
            B = (B + B.T) / 2

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.sum(B[np.tile(ci, (n, 1)) == np.tile(ci, (n, 1)).T]) / s

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                raise BCTParamError('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # pool weights of nodes in same module
                bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
                b1[i - 1, j - 1] = bm
                b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q
        q = np.trace(B) / s  # compute modularity

    return ci, q


def multislice_static_unsigned(A, g_plus=1):
    """
    Inputs
    ======
    A: (weighted) connectivity matrix
    it is assumsed that all values of the connectivity matrix are positive
    g_plus: the resolution parameter. If unsure, use default value of 1.

    Outputs
    =======
    S: the partition (or community assignment of all nodes to communities)
    Q: the modularity of the (optimal) partition
    lAlambda: the effective fraction of antiferromagnetic edges
        (see Onnela et al. 2011 http://arxiv.org/pdf/1006.5731v1.pdf)
    """

    Aplus = A
    Aplus[A < 0] = 0

    k_plus = np.sum(Aplus, axis=1)
    P = np.outer(k_plus, k_plus) / sum(k_plus)
    B = A - g_plus * P
    lAlambda = (np.divide(A,P) < gplus).sum()

    S, Q = community_louvain(B)
    Q = Q/sum(Aplus)

    return S, Q, lAlambda


def symm_matrix(A, binary=False):
    """
    A function to symmetrize an input matrix. The procedure
    by which items are symmetrized such that:
    in the binary case:
        if a(i,j) || a(j,i) == 1 for i,j in A
            w(i,j) && w(j,i) == 1
    in  the weighted case:
        if (a(i,j) || a(j,i) > 0 for i,j in A
            w(i,j) && w(j,i) == (a(i,j) + a(j,i) )/ 2

    Inputs
    ======
    A: np.array, the binary or weighted input matrix
    binary: bool, the key to indicate whether weighted or binary analysis
        will take place
        True indicates binarized, False indicates weighted

    Outputs
    =======
    W: The symmeterized matrix

    if binary analysis is specified, let binary symmeterization take place

    written by Eric W. Bridgeford
    """
    if binary:
        W = A.astype(int)
        for (i, j) in combinations(range(len(A)), 2):
            if A[i, j] or A[j, i]:
                W[i, j] = 1
                W[j, i] = 1
    else:
        W = (A + A.T) / 2.0

    return W


def regular_matrix_generator(G, r):
    """
    Generates a regular matrix, with weights obtained form the
    original adjacency matrix representation of the network

    note that all inputs should be symmeterized prior to forming a regular
    matrix, since otherwise half of the connnections will be trashed. This
    can be accomplished with the built in symm_matrix function, however,
    the function is not performed here so that users can use their own
    symmeterization procedure.

    Inputs
    ======
    G: the adjacency matrix for the given network; must be symmmeterized
    r: the approximate radius of the regular network

    Outputs
    =======
    M: the regular matrix for the given network, where all
        weights are sorted such that the inner radius has the
        highest weights randomly distributed across the nodes,
        and so on
    """
    n = len(G)
    r = int(r)
    B = np.triu(G).ravel()
    B = np.sort(B, axis=0)
    B = np.flip(B, axis=0)
    n_elems = np.ceil(len(G.ravel())/ (2*len(G)))
    n_zeros = 2 * n * n_elems - len(G.ravel())
    B = np.concatenate((B, np.zeros(n_zeros.astype(int))))
    B = np.reshape(B, (n, int(len(B)/n)), order='F')

    M = np.zeros((n, n))
    for i in range(n):
        for z in range(r):
            a = np.random.randint(0, n)
            while B[a, z] == 0 and z != r:
                a = np.random.randint(0, n)
            y_coor = np.mod(i + z - 1, n) + 1
            M[i, y_coor] = B[a, z]
            M[y_coor, i] = B[a, z]
            B[a, z] = 0
    return M


def randomize_matrix(A):
    """
    This code creates a random undirected network from the connectivity
    distribution of an undirected adjacency matrix, ie, the intital matrix
    must be symmetric.

    Inputs
    ======
    A: an undirected adjacency matrix (symmetric) with no self connections

    Outputs
    =======
    A_rand: a comparable random network with same number of nodes and
        connectivity distribution

    written by Sarah F. Muldoon
    """
    n_nodes = len(A)
    A_rand = np.zeros((n_nodes, n_nodes))

    orig_edges = A[np.triu_indices(len(A), k=1)]
    n_edges = len(orig_edges)

    randomized_edges = np.random.permutation(orig_edges)
    edge = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            A_rand[i, j] = randomized_edges[edge]
            A_rand[j, i] = randomized_edges[edge]
            edge = edge + 1
    return A_rand


def clustering_coef_matrix(W, method='O'):
    """
    A modification of the clustering coefficient function provided
    in the brain connectivity toolbox

    improved definition of Onnela Clustering Coefficient, as well as
    implementation of function for Zhang and Barrat clustering values

    Reference:
    - Onnela et al., Phys. Rev. E71, 065103(R)(2005)
    - B.Zhang and S. Horvath, Stat. App. Genet. Mol. Biol.4, 17(2005)
    - Barrat et al., Proc. Natl. Acad. Sci. U.S.A.101, 3747(2004)
    - Watts and Strogatz (1998) Nature 393:440-442

    Inputs
    ======
    W: the weighted or unweighted connectivity matrix
    method: str, to represent the method to be used for computing
        the clustering coefficient
        possible strings: 'O' (Onnela), 'Z' (Zhang), 'B' (Barrat), 'bin' (binary)

    """

    if method == 'O':
        K = np.sum(W, axis=1).astype(float)
        W2 = W / W.max()
        cyc3 = np.diag((W2 ** 1/3) ** 3)
        K[cyc3 == 0] = np.inf
        C = cyc3 / (K * (K - 1))
        return C

    elif method == 'bin':
        G = (W > 0).astype(float)
        n = len(G)
        C = []
        for u in range(n):
            V = np.where(G[u, :])[0]
            k = len(V)
            if k >= 2:
                S = G[V[:, None], V]
                C.append(S.sum() / (k**2 / k))
        return np.array(C)

    elif method == 'Z':
        C = []
        n = len(W)
        K = np.sum(W, axis=1)
        W = W.astype(float)
        W2 = W / W.max()
        cyc3 = np.diag(W2 ** 3)
        for i in range(n):
            denom = np.sum(W2[i, :]) ** 2 - np.sum(W2[i, :] ** 2)
            C.append(denom)
        return np.array(C)

    elif method == 'B':
        C = []
        n = len(G)
        G = (W > 0).astype(float)
        for i in range(n):
            sum1 = 0
            for j in range(n):
                for k in range(n):
                    sum1 = sum1 + ((W[i, j] + W[i, k]) / 2.) * A[i, j] * A[j, k] * A[i, k]
            c_i = sum1 / (np.sum(W[i, :]) * np.sum(G[i, :]))
            C.append(c_i)
        return np.array(C)

    else:
        print("Select methods from 'O', 'bin', 'Z' or 'B'")


def avg_clus_matrix(M, method='O'):
    """
    A function to compute the average clusteirng coefficient for a
    input matrix M

    Inputs
    ======
    W: a matrix, weighted or unweighted
    method: a string, to represent the method to be used for computing
        the clustering coefficient
        possible strings: 'O' (Onnela), 'Z' (Zhang), 'B' (Barrat), 'bin' (binary)

    Outputs
    =======
    Clus  the average clustering coefficient
    """
    C = clustering_coef_matrix(M, method)
    return np.nanmean(C)


def avg_path_matrix(M):
    """
    A function to compute the average path length of a given matrix
    using the graphallshortestpaths built-in matlab function

    written by Eric Bridgeford
    """
    M = np.array(M)
    r, c = np.tril_indices(len(M))
    G = nx.Graph()
    n = len(M)
    for r_, c_ in zip(r, c):
        if not np.isinf(M[r_, c_]):
            G.add_edge(r_, c_, weight=M[r_, c_])
    return nx.average_shortest_path_length(G)


def small_world_propensity(A, method='O'):
    """
    A function for calculating the small world propensity of
    a given network - assumes that matrix is undirected (symmeteric) and if
    not, creates a symmetric matrix which is used for the calculations

    Inputs
    ======
    A: np.array, the connectivity matrix, weighted or binary
    method: str, a string corresponding to the method of clustering
        to be used, where 'O' is Onnela, 'Z' is Zhang,
        'B' is Barrat, 'bin' is binary (default is Onnela).
        If the user specifies binary analysis, a
        weighted matrix will be converted to a binary matrix
        before proceeding.

    Outputs
    =======
    SWP: the small world propensity of the matrix
    delta_C: the fractional deviation from the expected culstering coefficient of a
        random network
    delta_L: the fractional deviation from the expected path length of a
        random network

    written by Eric Bridgeford and modified by Sarah F. Muldoon

    Reference:
        Muldoon, Bridgeford, and Bassett (2015) "Small-World Propensity in Weighted,
        Real-World Networks" http://arxiv.org/abs/1505.02194
    """
    if A.sum() > 0:
        bin_matrix = False
        if method == 'bin':
            bin_matrix = True
            A = (A > 0).astype(int)
        sym_check = np.abs(A - A.T).sum()
        if sym_check > 0:
            W = symm_matrix(A, False)
        else:
            W = A
        n = len(W)
        dens_net = W.sum()/(W.max() * n * (n - 1))
        n_connections = (W > 0).sum()
        avg_deg_unw = n_connections / n
        avg_rad_unw = avg_deg_unw / 2.
        avg_rad_eff = np.ceil(avg_rad_unw)

        # compute the average degree of the unweighted network, to give
        # the approximate radius
        W_reg = regular_matrix_generator(W, avg_rad_eff)
        W_rand = randomize_matrix(W)

        # compute all path length calculations for the network
        reg_path = avg_path_matrix(1 / W_reg) # path of the regular network
        rand_path = avg_path_matrix(1 / W_rand) # path of the random netowork
        net_path = avg_path_matrix(1 / W) # path of the network

        A = net_path - rand_path
        if A < 0:
            A = 0
        diff_path = A/ (reg_path - rand_path)

        reg_clus = avg_clus_matrix(W_reg, method)
        rand_clus = avg_clus_matrix(W_rand, method)
        net_clus = avg_clus_matrix(W, method)

        B = (reg_clus - net_clus)
        if B < 0:
            B = 0
        diff_clus = B / (reg_clus - rand_clus)
        if np.isnan(reg_clus) or np.isnan(rand_clus) or np.isnan(net_clus):
            diff_clus = 1
        if diff_clus > 1:
            diff_clus = 1

        SWP = 1 - (np.sqrt(diff_clus ** 2 + diff_path ** 2)/np.sqrt(2))
        delta_C = diff_clus
        delta_L = diff_path

        return SWP, delta_C, delta_L
