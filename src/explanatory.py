#%%
import pandas as pd
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
np.random.seed(42)

def plot_stats(g):
    # in & out degree distribution
    in_hist = gt.vertex_hist(g, "in")
    out_hist = gt.vertex_hist(g, "out")

    y = in_hist[0]
    err = np.sqrt(in_hist[0])
    plt.errorbar(in_hist[1][:-1], in_hist[0], fmt="o", yerr=err,
            label="in", alpha=0.5)

    y = out_hist[0]
    err = np.sqrt(out_hist[0])
    plt.errorbar(out_hist[1][:-1], out_hist[0], fmt="o", yerr=err,
            label="out", alpha=0.5)

    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("$k$")
    plt.ylabel("$NP_k$")

    #plt.tight_layout()
    plt.legend();

    avg_dgr = gt.vertex_average(g, "total")

    print(f"Average degree (total): {avg_dgr}")

def plot_stats_in(g):
    # in & out degree distribution
    in_hist = gt.vertex_hist(g, "in")

    y = in_hist[0]
    err = np.sqrt(in_hist[0])
    plt.errorbar(in_hist[1][:-1], in_hist[0], fmt="o", yerr=err,
            label="in", alpha=0.5)


    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("$k$")
    plt.ylabel("$NP_k (in)$")

    #plt.tight_layout()
    plt.legend();

    avg_dgr = gt.vertex_average(g, "total")

    print(f"Average degree (total): {avg_dgr}")

def plot_bar(x, y, title=None, xtick=None, rotation=0, xlabel=None, ylabel=None, width=0.8, horizontal=False, legend_label=None):
    if horizontal:
        plt.barh(y=x, width=y, height=width, alpha=0.5, label=legend_label)
        xlabel, ylabel = ylabel, xlabel
        plt.gca().invert_yaxis()
    else:
        plt.bar(x=x, height=y, width=width, alpha=0.5, label=legend_label)
    plt.xticks(xtick, rotation=rotation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# PRPEARE DATA
dfo = pd.read_parquet("data/processed/retweets.parquet")

# OBSERVED GRAPH
df = dfo.copy()
df = df.dropna(subset=["retweet_created_at"])
df["day"] = df.retweet_created_at.dt.date

# choose columns to keep
columns = ['retweet_author_username',
    'tweet_author_username',
    "day",
    ]

df = df[columns].reset_index(drop=True)

# get information about the retweets
def get_time_range(df):
    df.retweet_created_at = df.retweet_created_at.astype("datetime64[ns, UTC]")
    start_date = df.retweet_created_at.min().strftime("%B %e, %Y")
    end_date = df.retweet_created_at.max().strftime("%B %e, %Y")

    return start_date, end_date

dfo = dfo.dropna(subset=["retweet_created_at"])
start_date, end_date = get_time_range(dfo)

print(f"\nTime range of the retweets:\n{start_date} - {end_date}")

T = df.day.nunique()
N = len(set(list(df.retweet_author_username) + list(df.tweet_author_username)))
L = df.shape[0]

# # find out average number of new retweets per user per day
# # this is the average number of edges per user per time step
# avg_num_edges = df.groupby(["retweet_author_username", "day"]).size().mean()

# calculate the average number of new retweets per day
avg_num_edges_day = df.groupby(["day"]).size().mean()

# calculate average number of users retweeting per day
avg_num_users_day = df.groupby(["retweet_author_username", "day"], as_index=False).size()
avg_num_users_day = avg_num_users_day.groupby(["day"]).size().mean()

# activity value: average number of retweets per user per day (considering all users each day)
act_pot = avg_num_edges_day / N
print(f"Calculated activity value: {act_pot}")

# Get user-specific activity values
df1 = df.groupby(["retweet_author_username"], as_index=False).size()
df1["avg_hourly_activity"] = df1["size"] / (71 * 24)

df = df.merge(df1[["retweet_author_username", "avg_hourly_activity"]], on="retweet_author_username", how="left")

#%% CREATE GRAPHS

# OBSERVED
g_list = [(r, t) for r, t in zip(df.retweet_author_username, df.tweet_author_username)]
gof = gt.Graph(
    g_list,
    hashed=True,
    directed=True,
    )

# GENERATED (RANDOM)
ggf = gt.Graph(directed=True)

# add vertices
for i in range(N):
    ggf.add_vertex()

# add edges
targets = list(range(N))
for t in range(T):
    for i in range(N):
        if np.random.rand() < act_pot:
            poss_targets = targets.copy().remove(i) # remove self-loops
            receiver = np.random.choice(poss_targets) # choose random target
            ggf.add_edge(i, receiver)

#%%

# GENERATED (USER-SPECIFIC)
# create list of users and their activity values
df_ = df.copy()
df_ = df_[["retweet_author_username", "avg_hourly_activity"]].drop_duplicates()
avg_hourly_activity = list(df_["avg_hourly_activity"].values)
while len(avg_hourly_activity) < N:
    avg_hourly_activity.append(0)

# create graph
ggf2 = gt.Graph(directed=True)

# add vertices
for i in range(N):
    ggf2.add_vertex()

# add edges
targets = list(range(N))
T_ = T*24
for t in range(T_):
    for i, act in enumerate(avg_hourly_activity):
        if np.random.rand() < act:
            poss_targets = list(filter(lambda x: x != i, targets)) # remove self-loops
            receiver = np.random.choice(poss_targets) # choose random target
            ggf2.add_edge(i, receiver)


#%% GENERATED (TRIADIC CLOSURE)

def gt_get_random_neighbour(graph, node, exceptions=None):
    if exceptions is None:
        exceptions = set()

    neighbours = [neighbour for neighbour in node.out_neighbours()
                  if neighbour not in exceptions]
    # total_weight = sum(graph.ep.weight[graph.edge(node, neighbour)]
    #                    for neighbour in neighbours)
    # weight_dist = [graph.ep.weight[graph.edge(node, neighbour)] / total_weight
    #                for neighbour in neighbours]
    #return np.random.choice(neighbours, p=weight_dist)
    return np.random.choice(neighbours)


def gt_focal_closure(graph, active_node, exceptions=None):
    '''
    A new edge is formed between the active node and a random node
    that is not already a neighbor of the active node
    and is not one of the exceptions provided (i.e., exceptions argument).
    '''
    if exceptions is None:
        exceptions = set()
    # exceptions should include the active node and its neighbours
    exceptions |= {active_node}
    exceptions |= set(active_node.out_neighbours())

    # it should be more efficient for large graphs to potentially redraw a node
    # index instead of explicity building a list of nodes that excludes the
    # exceptions and draw from it
    while True:
        node_idx = np.random.randint(graph.num_vertices())
        other_node = graph.vertex(node_idx)
        if other_node not in exceptions:
            break

    edge = graph.add_edge(active_node, other_node)
    #graph.ep.weight[edge] = 1.0
    return other_node

def gt_triadic_closure(graph, active_node):
    '''
    Performs a triadic closure on the graph by adding an edge between the
    active node and a random neighbour of a random neighbour of the active
    node.

    Parameters
    ----------
    graph : gt.Graph
        The graph to perform the triadic closure on.
    active_node : gt.Vertex
        The node to perform the triadic closure on.

    Returns
    -------
    other_node : gt.Vertex
        The node that was chosen as the other node in the triadic closure.
    '''
    neighbour = gt_get_random_neighbour(graph, active_node)
    if neighbour.out_degree() == 1:
        return gt_focal_closure(graph, active_node)

    other_node = gt_get_random_neighbour(graph, neighbour,
                                         exceptions={active_node})
    # if other_node in active_node.out_neighbours():
    #     edge = graph.edge(active_node, other_node)
    #     graph.ep.weight[edge] += graph.gp.link_reinforecement
    #     return other_node

    if np.random.rand() < graph.gp.p_triadic_closure:
        edge = graph.add_edge(active_node, other_node)
        #graph.ep.weight[edge] = 1.0
        return other_node

    return gt_focal_closure(graph, active_node, exceptions={other_node})


# create graph
ggf3 = gt.Graph(directed=True)

# add vertices
for i in range(N):
    ggf3.add_vertex()
graph = ggf3.copy()

# add edges
memory_strength = 1.0 # higher values = more likely to connect to non-neighbour nodes
targets = list(range(N))
T_ = T*24
p_triadic_closure = 0.75
for t in range(T_):
    for i, act in enumerate(avg_hourly_activity):
        if np.random.rand() < act:
            node = graph.vertex(i)

            if node.out_degree() == 0:
                # connect to random node
                poss_targets = list(filter(lambda x: x != i, targets)) # remove self-loops
                receiver = np.random.choice(poss_targets) # choose random target
                graph.add_edge(i, receiver)
            else:
                # connect to neighbour of neighbour
                neighbour = gt_get_random_neighbour(graph, node, exceptions={node})
                if neighbour.out_degree() <= 1:
                    gt_focal_closure(graph, node) # connect to random node
                else:
                    other_node = gt_get_random_neighbour(graph, neighbour, exceptions={node})
                    if other_node in node.out_neighbours():
                        # if other_node is already a neighbour, connect to it
                        graph.add_edge(node, other_node)
                    elif np.random.rand() < p_triadic_closure:
                        graph.add_edge(node, other_node)
                    else:
                        gt_focal_closure(graph, node) # connect to random node


ggf3 = graph.copy()

            # poss_targets = list(filter(lambda x: x != i, targets)) # remove self-loops
            # receiver = np.random.choice(poss_targets) # choose random target
            # ggf2.add_edge(i, receiver)


            # p_new_tie = memory_strength / (memory_strength + node.out_degree())
            # if np.random.rand() < p_new_tie:
            #     if node.out_degree() == 0:
            #         # connect to random node
            #         poss_targets = list(filter(lambda x: x != i, targets)) # remove self-loops
            #         receiver = np.random.choice(poss_targets) # choose random target
            #         ggf2.add_edge(i, receiver)
            #     else:
            #         # connect to neighbour of neighbour
            #         #other_node = gt_triadic_closure(ggf3, node)
            #         neighbour = gt_get_random_neighbour(ggf3, node)
            #         if neighbour.out_degree() == 1:
            #             # connect to random node that is not self or neighbour
            #             poss_targets = list(filter(lambda x, j: x != i and x not in set(list(ggf3.get_out_neighbours(i)), targets)))
            #             receiver = np.random.choice(poss_targets) # choose random target
            #             ggf2.add_edge(i, receiver)
            #         else:
            #             other_node = gt_get_random_neighbour(graph, neighbour,
            #                              exceptions={active_node})

#%% PREFERENTIAL ATTACHMENT

def random_link(graph, i, targets):
    # connect to random node
    poss_targets = list(filter(lambda x: x != i, targets)) # remove self-loops
    receiver = np.random.choice(poss_targets) # choose random target
    graph.add_edge(i, receiver)


def random_link_preferential(graph, i):
    # connect to a node based on preferential attachment
    nodes = [node for node in graph.vertices()]
    degrees = [node.out_degree() for node in nodes]
    degrees[i] = 0 # remove self-degree
    total_degree = sum(degrees)
    probs = [degree / total_degree for degree in degrees]
    receiver = np.random.choice(nodes, p=probs)
    graph.add_edge(i, receiver)

def random_link_preferential(graph, i):
    # connect to a node based on preferential attachment
    try:
        nodes = [node for node in graph.vertices()]
        degrees = [node.out_degree() for node in nodes]
        degrees[i] = 0 # remove self-degree
        total_degree = sum(degrees)
        probs = [degree / total_degree for degree in degrees]
        receiver = np.random.choice(nodes, p=probs)
        graph.add_edge(i, receiver)
    except ZeroDivisionError:
        nodes = [node for node in ggf4_.vertices()]
        degrees = [node.out_degree() for node in nodes]
        degrees[i] = 0 # remove self-degree
        total_degree = sum(degrees)
        probs = [degree / total_degree for degree in degrees]
        receiver = np.random.choice(nodes, p=probs)
        graph.add_edge(i, receiver)


def gt_get_random_neighbour_pref(graph, node, exceptions=None):
    neighbours = [neighbour for neighbour in node.out_neighbours()]
    total_degree = sum(neighbour.out_degree() for neighbour in neighbours)
    probs = [neighbour.out_degree() / total_degree for neighbour in neighbours]
    return np.random.choice(neighbours, p=probs)

def gt_get_random_neighbour_pref(graph, node, exceptions=None):
    if exceptions is None:
        exceptions = set()

    try:
        neighbours = [neighbour for neighbour in node.out_neighbours()]
        total_degree = sum(neighbour.out_degree() for neighbour in neighbours)
        probs = [neighbour.out_degree() / total_degree for neighbour in neighbours]
        return np.random.choice(neighbours, p=probs)
    except ZeroDivisionError:
        neighbours = [neighbour for neighbour in node.out_neighbours()]
        return np.random.choice(neighbours)

def gt_get_random_neighbour(graph, node, exceptions=None):
    if exceptions is None:
        exceptions = set()

    neighbours = [neighbour for neighbour in node.out_neighbours()]
    return np.random.choice(neighbours)

# create graph
ggf4 = gt.Graph(directed=True)

# add vertices
for i in range(N):
    ggf4.add_vertex()
graph = ggf4.copy()

# add edges
targets = list(range(N))
# for i, act in enumerate(avg_hourly_activity):
#     if np.random.rand() < act:
#         targets.append(i)

targets = list(range(N))
ggf4_ = gt.Graph(directed=True)
for i in range(N):
    ggf4_.add_vertex()
for t in range(24):
    for i, act in enumerate(avg_hourly_activity):
        if np.random.rand() < act:
            random_link(ggf4_, i, targets)


T_ = T*24
p_triadic_closure = 0.75
for t in range(T_):
    for i, act in enumerate(avg_hourly_activity):
        if np.random.rand() < act:
            node = graph.vertex(i)

            if node.out_degree() == 0:
                random_link_preferential(graph, i)

            else:
                # connect to neighbour of neighbour
                neighbour = gt_get_random_neighbour_pref(graph, node, exceptions={node})
                if neighbour.out_degree() <= 1:
                    random_link_preferential(graph, i)
                else:
                    other_node = gt_get_random_neighbour_pref(graph, neighbour, exceptions={node})
                    if other_node in node.out_neighbours():
                        # if other_node is already a neighbour, connect to it
                        graph.add_edge(node, other_node)
                    elif np.random.rand() < p_triadic_closure:
                        graph.add_edge(node, other_node)
                    else:
                        random_link_preferential(graph, i)

            #targets.append(i)

ggf4 = graph.copy()

#############################################################################################
#%%


#%%
def plot_full_graph(g, output_path):
    pos = gt.sfdp_layout(ggf)
    gt.graph_draw(ggf, pos=pos, output=output_path)

    return pos, g

pos_ggf, ggf = plot_full_graph(ggf4, "figures/gen4_closure_full.png")

def plot_lc(g, output_path):
    g_ = gt.extract_largest_component(g, directed=False, prune=True)
    pos = gt.sfdp_layout(g_)
    gt.graph_draw(g_, pos=pos, output=output_path)

    return pos, g_

pos_gg, gg = plot_lc(ggf4, "figures/gen4_closure_lc.png")

###############################################################################################

#%% COMPARE GRAPHS

# compare number of edges
print("STATS FOR FULL GRAPHS")
gg_target = ggf
print(f"Edges \nobserved graph: {gof.num_edges()}\ngenerated graph: {gg_target.num_edges()} ({round((gg_target.num_edges() / gof.num_edges())*100,1)}%)")
print(f"\nVertices \nobserved graph: {gof.num_vertices()}\ngenerated graph: {gg_target.num_vertices()} ({round((gg_target.num_vertices() / gof.num_vertices())*100,1)}%)")

# print(f"Number of vertices in observed graph: {gof.num_vertices()}")
# print(f"Number of vertices in generated graph: {ggf2.num_vertices()}")

# compare number of edges
print("\nSTATS FOR LARGEST COMPONENT")
gg_target = gg
print(f"Edges \nobserved graph: {go.num_edges()}\ngenerated graph: {gg_target.num_edges()} ({round((gg_target.num_edges() / go.num_edges())*100,1)}%)")
print(f"\nVertices \nobserved graph: {go.num_vertices()}\ngenerated graph: {gg_target.num_vertices()} ({round((gg_target.num_vertices() / go.num_vertices())*100,1)}%)")

#%% CLUSTERING COEFFICIENT
print("GLOBAL CLUSTERING COEFFICIENT")
print(f"Observed larg. comp. = {round(gt.global_clustering(go)[0],5)} (std = {round(gt.global_clustering(go)[1],5)})")
print(f"Generated larg. comp. = {round(gt.global_clustering(gg)[0],5)} (std = {round(gt.global_clustering(gg)[1],5)})")

lcc_o = list(gt.local_clustering(go))
lcc_g = list(gt.local_clustering(gg))

print("\nLOCAL CLUSTERING COEFFICIENT")
print(f"Observed larg. comp. (average) = {round(np.mean(lcc_o),5)} (std = {round(np.std(lcc_o),5)})")
print(f"Generated larg. comp. (average) = {round(np.mean(lcc_g),5)} (std = {round(np.std(lcc_g),5)})")

# %% DEGREE DISTRIBUTION (OBSERVED)
plot_stats(go)
plt.title("Degree distribution for largest component")
plt.savefig("figures/go_degree_dist.png")


#%% DEGREE DISTRIBUTION (BOTH)
plot_stats(go)
plot_stats(gg)
plt.legend(['in (observed)', 'out(observed)', 'in (generated)', 'out (generated)'])
plt.title("Degree distribution for largest components")
plt.savefig("figures/go+gg2_degree_dist.png")

#%% IN-DEGREE DISTRIBUTION (BOTH)
plot_stats_in(go)
plot_stats_in(gg)
plt.legend(['in (observed)', 'in (generated)'])
plt.title("In-degree distribution for largest components")
plt.savefig("figures/go+gg_in_degree_dist.png")

#%% HIGHEST DEGREE VERTICES
def highest_degree_vertices(g, n=10):
    in_degrees = g.get_in_degrees(range(g.num_vertices()))
    out_degrees = g.get_out_degrees(range(g.num_vertices()))

    in_degrees = pd.Series(in_degrees).sort_values(ascending=False).head(n)
    out_degrees = pd.Series(out_degrees).sort_values(ascending=False).head(n)

    # set name of index and column
    #in_degrees.index.name = "Vertex"
    # add column name

    in_degrees = pd.DataFrame(in_degrees).reset_index()
    in_degrees.columns = [["Vertex", "In-degree"]]

    out_degrees = pd.DataFrame(out_degrees).reset_index()
    out_degrees.columns = [["Vertex", "Out-degree"]]

    print(in_degrees)
    print("\n")
    print(out_degrees)

highest_degree_vertices(go)
highest_degree_vertices(gg)


# %%

def weight_dist(plot_bar, go):
    edges_go = pd.DataFrame(go.get_edges())
    edges_go = edges_go.groupby([0, 1], as_index=False).size().reset_index(drop=True)
    edges_go.rename(columns={0: "Retweeter", 1: "Tweeter", "size": "weight"}, inplace=True)

    edges_weight_count = edges_go.weight.value_counts().sort_index()

    plot_bar(x=edges_weight_count.index, y=edges_weight_count.values,
    title="Edges weight distribution",
    xtick=None, rotation=0,
    xlabel="Weight (i.e. number of edges between vertex pairs)", ylabel="Edges count");

    return edges_weight_count

weights_go = weight_dist(plot_bar, go)
plt.savefig("figures/go_weight_dist.png")

# %%

def weight_dist(plot_bar, go):
    edges_go = pd.DataFrame(go.get_edges())
    edges_go = edges_go.groupby([0, 1], as_index=False).size().reset_index(drop=True)
    edges_go.rename(columns={0: "Retweeter", 1: "Tweeter", "size": "weight"}, inplace=True)

    edges_weight_count = edges_go.weight.value_counts().sort_index()
    print(edges_weight_count)

    plot_bar(x=edges_weight_count.index, y=edges_weight_count.values,
    title="Edges weight distribution",
    xtick=None, rotation=0,
    xlabel="Weight (i.e. number of edges between vertex pairs)", ylabel="Edges count");

    return edges_weight_count

weights_gg = weight_dist(plot_bar, gg)

# %%
plt.bar(x=weights_go.index, height=weights_go.values, alpha=0.5, label="observed")
#plt.bar(x=weights_gg.index, height=weights_gg.values, alpha=0.5, label="generated")
plt.bar(x=weights_gg.head(29).index, height=weights_gg.head(29).values, alpha=0.5, label="generated")
plt.legend()
plt.title("Edges weight distribution (largest components)")
plt.xlabel("Weight (i.e. number of edges between vertex pairs)")
plt.ylabel("Edges count")
plt.savefig("figures/go+gg_weight_dist.png")

# %% SAVE GRAPHS

gof.save("data/processed/gof.gt.gz")
go.save("data/processed/go.gt.gz")

ggf.save("data/processed/ggf.gt.gz")
gg.save("data/processed/gg.gt.gz")

ggf2.save("data/processed/ggf2.gt.gz")
gg2.save("data/processed/gg2.gt.gz")

#%% LOAD GRAPHS
gof = gt.load_graph("data/processed/gof.gt.gz")
go = gt.load_graph("data/processed/go.gt.gz")

ggf = gt.load_graph("data/processed/ggf.gt.gz")
gg = gt.load_graph("data/processed/gg.gt.gz")

ggf2 = gt.load_graph("data/processed/ggf2.gt.gz")
gg2 = gt.load_graph("data/processed/gg2.gt.gz")

# %%
