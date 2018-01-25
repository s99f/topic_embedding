__author__ = 'ando'
import os,sys
import random
from multiprocessing import cpu_count
import logging as log


import numpy as np
import psutil
from math import floor
from ADSCModel.model import Model
from ADSCModel.context_embeddings import Context2Vec
from ADSCModel.node_embeddings import Node2Vec
from ADSCModel.community_embeddings import Community2Vec
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import timeit
import tensorflow as tf
import scipy.spatial.distance as distance

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

epslion = 1e-6


p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

def simi(G, model, iter, nodeid2cluster):
    in_cluster = 0.0
    in_cluster_emb = 0.0
    in_cluster_num = 0.0
    out_cluster = 0.0
    out_cluster_emb = 0.0
    out_cluster_num = 0.0
    #K=1
    #com_learner.predict(G.nodes(), model)
    ##log.info('probility = {}'.format(model.probility[:3,:5]))
    #for i in range(model.probability.shape[0]):
    #    max_idx = np.argpartition(model.probability[i],-K)[-K:]
    #    for j in range(len(max_idx)):
    #        if max_idx[j] not in cluster2nodeid:
    #            cluster2nodeid[max_idx[j]] = []
    #        cluster2nodeid[max_idx[j]].append(model.vocab_t[i])
    #        if model.vocab_t[i] not in nodeid2cluster:
    #            nodeid2cluster[model.vocab_t[i]] = max_idx[j]
    i = 0
    swing_d = {}
    emb_d = {}
    for item in G.edges():
        #if i < 3:
        #    print('weight = ', G[item[0]][item[1]]['weight'])
        #    i+=1
        swing_d[item] = G[item[0]][item[1]]['weight']
        emb_d[item] = np.dot(model.node_embedding[model.vocab[item[0]].index],
                                        model.node_embedding[model.vocab[item[1]].index])
        if nodeid2cluster[item[0]] != nodeid2cluster[item[1]]:
            out_cluster_num += 1
            out_cluster += G[item[0]][item[1]]['weight']
            #out_cluster_emb += distance.cosine(model.node_embedding[model.vocab[item[0]].index],
            #                            model.node_embedding[model.vocab[item[1]].index])
            out_cluster_emb += np.dot(model.node_embedding[model.vocab[item[0]].index],
                                        model.node_embedding[model.vocab[item[1]].index])
        else:
            in_cluster_num += 1
            in_cluster += G[item[0]][item[1]]['weight']
            #in_cluster_emb += distance.cosine(model.node_embedding[model.vocab[item[0]].index],
            #                            model.node_embedding[model.vocab[item[1]].index])
            in_cluster_emb += np.dot(model.node_embedding[model.vocab[item[0]].index],
                                        model.node_embedding[model.vocab[item[1]].index])
    #log.info('in_cluster_num = {}'.format(in_cluster_num))
    #log.info('out_cluster_num = {}'.format(out_cluster_num))
    log.info('iter = %d, in_cluster_num = %f, in_cluster = %f, out_cluster_num = %f, out_cluster=%f, \
            in_ratio = %f, out_ratio = %f, in_emb_ratio = %f, out_emb_ratio = %f' \
            % (iter, in_cluster_num, in_cluster, out_cluster_num, out_cluster,
                in_cluster/(in_cluster_num+epslion), out_cluster/(out_cluster_num+epslion),
                in_cluster_emb/(in_cluster_num+epslion), out_cluster_emb/(out_cluster_num+epslion)))
    #swing_l = sorted(swing_d.items(), key=lambda x: x[1], reverse=True)
    #emb_l = sorted(emb_d.items(), key=lambda x: x[1], reverse=True)
    #swing_l_t = swing_l[:int(len(swing_l)*0.2)]
    #emb_l_t = emb_l[:int(len(emb_l)*0.2)]
    #score2num = {}
    #for item in swing_l_t:
    #    if item[1] not in score2num:
    #        score2num[item[1]] = 0
    #    score2num[item[1]] += 1
    #score2num_l = sorted(score2num.items(), key=lambda x: x[1], reverse=True)
    #log.info('score2num top = {}, total = {}'.format(score2num_l[:5], len(score2num)))
    #log.info('swing_l = {}, emb_l = {}'.format(swing_l, emb_l))
    #swing_s = set([ item[0] for item in swing_l_t ] )
    #emb_s = set([ item[0] for item in emb_l_t] )
    #log.info('total = %d, intersection = %d, ratio = %f' 
    #        % (len(swing_s), len(swing_s.intersection(emb_s)), 
    #        len(swing_s.intersection(emb_s))/len(swing_s) ))

if __name__ == "__main__":

    #Reading the input parameters form the configuration files
    number_walks = 20                       # number of walks for each node
    walk_length = 80                        # length of each walk
    representation_size = 16               # size of the embedding
    num_workers = 8                        # number of thread
    num_iter = 50                            # number of overall iteration
    reg_covar = 0.000001                     # regularization coefficient to ensure positive covar
    #reg_covar = 1e-7                     # regularization coefficient to ensure positive covar
    input_file = 'lazada_my_test'                # name of the input file
    #input_file = 'mini_test'                # name of the input file
    output_file = 'lazada_my_test'               # name of the output file
    #output_file = 'mini_test'               # name of the output file
    batch_size = 3000
    window_size = 5    # windows size used to compute the context embedding
    negative = 5        # number of negative sample
    negative4o1 = 5        # number of negative sample
    #lr = 0.025            # learning rate
    lr = 0.025            # learning rate

    
    alpha_betas = [(0.0, 0.1)]
    down_sampling = 0.0

    ks = [160]
    walks_filebase = os.path.join('data', output_file)            # where read/write the sampled path


    #CONSTRUCT THE GRAPH  nodeid not coding
    #G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    G = graph_utils.load_txtfile(os.path.join('./data', input_file, input_file + '.txt'), undirected=True)
    # Sampling the random walks for context
    log.info("sampling the paths")
    #walk_files = graph_utils.write_walks_to_disk(G, os.path.join(walks_filebase, "{}.walks".format(output_file)),
    #                                             num_paths=number_walks,
    #                                             path_length=walk_length,
    #                                             alpha=0,
    #                                             rand=random.Random(0),
    #                                             num_workers=num_workers)
    #walk_files = ['data/lazada_my/lazada_my.walks.0', 'data/lazada_my/lazada_my.walks.1',
    #             'data/lazada_my/lazada_my.walks.2', 'data/lazada_my/lazada_my.walks.3',
    #              'data/lazada_my/lazada_my.walks.4']
    #walk_files = ['data/lazada/lazada.walks.0', 'data/lazada/lazada.walks.1',
    #             'data/lazada/lazada.walks.2', 'data/lazada/lazada.walks.3',
    #              'data/lazada/lazada.walks.4']
    #walk_files = ['data/Dblp/Dblp.walks.0', 'data/Dblp/Dblp.walks.1', 'data/Dblp/Dblp.walks.2', 'data/Dblp/Dblp.walks.3', 'data/Dblp/Dblp.walks.4']
    walk_files = ['data/lazada_my_test/lazada_my_test.walks.0', 'data/lazada_my_test/lazada_my_test.walks.1', 'data/lazada_my_test/lazada_my_test.walks.2', 
    'data/lazada_my_test/lazada_my_test.walks.3', 'data/lazada_my_test/lazada_my_test.walks.4', 'data/lazada_my_test/lazada_my_test.walks.5', 
    'data/lazada_my_test/lazada_my_test.walks.6']
    #walk_files = ['data/lazada_my_test/lazada_my_test.walks.0']
    #walk_files = ['data/mini_test/mini_test.walks.0', 'data/mini_test/mini_test.walks.1', 'data/mini_test/mini_test.walks.2', 'data/mini_test/mini_test.walks.3'
    #, 'data/mini_test/mini_test.walks.4', 'data/mini_test/mini_test.walks.5', 'data/mini_test/mini_test.walks.6']
    #walk_files = ['data/lazada_my_v1/lazada_my_v1.walks.0', 'data/lazada_my_v1/lazada_my_v1.walks.1', 'data/lazada_my_v1/lazada_my_v1.walks.2', 'data/lazada_my_v1/lazada_my_v1.walks.3', 'data/lazada_my_v1/lazada_my_v1.walks.4']
    log.info("walk_files = {}".format(walk_files))

    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers) #dict 记录节点和次数
    #print(vertex_counts)
    
    model = Model(vertex_counts,
                  size=representation_size,
                  down_sampling=down_sampling,
                  table_size=100000000,
                  #table_size=100,
                  input_file=os.path.join(input_file, input_file),
                  path_labels="./data",
                  walk_files=walk_files)
    #print("connected path = ", model.connected_path)
    #sys.exit(1)
    #print("self.vocab = ", model.vocab)
    #print("self.vocab_t = ", model.vocab_t)
    #print("self.table = ", model.table)
    #sys.exit(1)
    #vocab is coding
    #print("node_embedding = ", model.node_embedding)

    #Learning algorithm
    #node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
    node_learner = Node2Vec(workers=num_workers, negative=negative4o1, lr=lr)
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
    #com_learner = Community2Vec(lr=lr)
    com_learner = Community2Vec(lr=0.0)

    node_learner.build_model_struct(len(model.vocab), model.layer1_size, G, model, lamda = 0.0, learning_rate=lr)

    #tensorflow init
    #config = tf.ConfigProto()
    config = tf.ConfigProto(device_count={"CPU": 2}, # limit to num_cpu_core CPU usage  
                inter_op_parallelism_threads = 2,   
                intra_op_parallelism_threads = 2,  
                log_device_placement=True)  
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver = tf.train.Saver()

    context_total_path = G.number_of_nodes() * number_walks * walk_length
    edges = np.array(G.edges())
    #edges = [(nbr, G[cur][nbr]['weight']) for nbr in G.neighbors(cur)]
    #log.info('edges = {}'.format(edges[:3]))
    log.info("context_total_path: %d" % (context_total_path))
    log.info('node total edges: %d' % G.number_of_edges())

    log.info('\n_______________________________________')
    log.info('\t\tPRE-TRAINING\n')
    ###########################
    #   PRE-TRAINING          #
    ###########################
    start_time = timeit.default_timer()
    #node_learner.train(model,
    #                   edges=edges,
    #                   G=G,
    #                   cluster_negtivate=False,
    #                   nodeid2cluster={},
    #                   iter=1,
    #                   chunksize=batch_size)
    #print("pre = ", node_learner)
    node_learner.train_reg_struct(sess, node_learner, 
                       model,
                       edges=edges,
                       G=G,
                       cluster_negtivate=False,
                       nodeid2cluster={},
                       iter=1,
                       chunksize=batch_size)
    log.info('pre node_learner train time: %.2fs' % (timeit.default_timer() - start_time))
    #start_time = timeit.default_timer()

    #cont_learner.train(model,
    #                   paths=graph_utils.combine_files_iter(walk_files),
    #                   total_nodes=context_total_path,
    #                   alpha=1,
    #                   chunksize=batch_size)
    #log.info('pre cont_learner train time: %.2fs' % (timeit.default_timer() - start_time))
    
    #model.save("{}_pre-training".format(output_file))

    ###########################
    #   EMBEDDING LEARNING    #
    ###########################
    iter_node = floor(context_total_path/G.number_of_edges())
    iter_com = floor(context_total_path/(G.number_of_edges()))
    log.info('iter_node: %d, iter_com: %d' % (iter_node, iter_com))
    # iter_com = 1
    # alpha, beta = alpha_betas

    for alpha, beta in alpha_betas:
        for k in ks:
            #model = model.load_model("{}_pre-training".format(output_file))
            model.reset_communities_weights(k)
            for it in range(num_iter):
                log.info('\n_______________________________________\n')
                log.info('\t\tITER-{}\n'.format(it))
                log.info('using alpha:{}\tbeta:{}\titer_com:{}\titer_node: {}'.format(alpha, beta, iter_com, iter_node))
                start_time = timeit.default_timer()

                #if it == 0:
                #    com_learner.fit(model, reg_covar=reg_covar, n_init=10)
                #else:
                #    com_learner.fit(model, reg_covar=reg_covar, n_init=10, 
                #                means_init=model.centroid)
                com_learner.fit(model, reg_covar=reg_covar, n_init=10)
                #centroid = np.zeros((k, representation_size), dtype=np.float32)
                #com_learner.fit(model, reg_covar=reg_covar, n_init=1, means_init=centroid)
                #log.info('com_learner.fit final!')
                #calc each node clusterid
                nodeid2cluster = {}
                K=1
                com_learner.predict(G.nodes(), model)
                #log.info('probility = {}'.format(model.probility[:3,:5]))
                for i in range(model.probability.shape[0]):
                    max_idx = np.argpartition(model.probability[i],-K)[-K:]
                    for j in range(len(max_idx)):
                        if model.vocab_t[i] not in nodeid2cluster:
                            nodeid2cluster[model.vocab_t[i]] = max_idx[j]
                simi(G, model, it, nodeid2cluster)

                #output embedding & probility
                #io_utils.save_embedding(model.node_embedding, model.vocab,
                #                    file_name="{}_alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_ds-{}".format(
                #                        output_file, alpha, beta, window_size, negative, lr, iter_com, iter_node, model.k, down_sampling))
                #io_utils.save_community(model.probability, model.vocab_t, file_name="{}_alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_ds-{}.pi".format(
                #                        output_file, alpha, beta, window_size, negative, lr, iter_com, iter_node, model.k, down_sampling))

                #node_learner.train(model,
                #                   edges=edges,
                #                   G=G,
                #                   cluster_negtivate=True,
                #                   nodeid2cluster=nodeid2cluster,
                #                   iter=iter_node,
                #                   chunksize=batch_size)
                node_learner.train_reg_struct(sess, node_learner, 
                                    model,
                                    edges=edges,
                                    G=G,
                                    cluster_negtivate=True,
                                    nodeid2cluster=nodeid2cluster,
                                    #iter=iter_node,
                                    iter = 1,
                                    chunksize=batch_size)

                loss = node_learner.loss(model, edges)
                log.info('node_learner loss:{}\n'.format(loss))
                log.info('node_learner.train final!')
                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
                start_time = timeit.default_timer()

                
                com_learner.train(G.nodes(), model, beta, chunksize=batch_size, iter=iter_com)

                loss = com_learner.loss(G.nodes(), model, beta, chunksize=batch_size)
                log.info('com_learner loss:{}\n'.format(loss))
                log.info('com_learner.train final!')
                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
                start_time = timeit.default_timer()

                cont_learner.train(model,
                                   paths=graph_utils.combine_files_iter(walk_files),
                                   total_nodes=context_total_path,
                                   alpha=alpha,
                                   chunksize=batch_size)

                loss = cont_learner.loss(model, graph_utils.combine_files_iter(walk_files),
                                         context_total_path, alpha=alpha)
                log.info('cont_learner loss:{}\n'.format(loss))

                log.info('time: %.2fs' % (timeit.default_timer() - start_time))
                
                # log.info(model.centroid)
            com_learner.fit(model, reg_covar=reg_covar, n_init=10)
                            #means_init=model.centroid)
            nodeid2cluster = {}
            K=1
            com_learner.predict(G.nodes(), model)
            #log.info('probility = {}'.format(model.probility[:3,:5]))
            for i in range(model.probability.shape[0]):
                max_idx = np.argpartition(model.probability[i],-K)[-K:]
                for j in range(len(max_idx)):
                    if model.vocab_t[i] not in nodeid2cluster:
                        nodeid2cluster[model.vocab_t[i]] = max_idx[j]
            simi(G, model, num_iter, nodeid2cluster)
            io_utils.save_embedding(model.node_embedding, model.vocab,
                                    file_name="{}_alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_ds-{}".format(
                                        output_file, alpha, beta, window_size, negative, lr, iter_com, iter_node, model.k, down_sampling))
            io_utils.save_community(model.probability, model.vocab_t, file_name="{}_alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_ds-{}.pi".format(
                                        output_file, alpha, beta, window_size, negative, lr, iter_com, iter_node, model.k, down_sampling))
