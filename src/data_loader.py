import collections
import os
import numpy as np


def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, nn, kg = load_graph(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, nn, ripple_set


def load_rating(args):
    rating_file = '../weibo' + '/retweet'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)


def dataset_split(rating_np):
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)
    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    return train_data, eval_data, test_data, user_history_dict


def load_graph(args):
    graph_file = '../weibo'  + '/follow'
    if os.path.exists(graph_file + '.npy'):
        graph_np = np.load(graph_file + '.npy')
    else:
        graph_np = np.loadtxt(graph_file + '.txt', dtype=np.int32)
        np.save(graph_file + '.npy', graph_np)

    n_entity = len(set(graph_np[:, 0]) | set(graph_np[:, 2]))
    n_relation = len(set(graph_np[:, 1]))

    graph = construct_graph(graph_np)

    return n_entity, n_relation, graph


def construct_graph(graph_np):
    graph = collections.defaultdict(list)
    for head, relation, tail in graph_np:
        graph[head].append((tail, relation))
    return graph


def get_ripple_set(args, graph, history_dict):

    ripple_set = collections.defaultdict(list)

    for user in history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in graph[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
