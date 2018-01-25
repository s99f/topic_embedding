__author__ = 'ando'

import logging as log
log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import chunkize_serial, RepeatCorpusNTimes, prepare_sentences
from scipy.special import expit as sigmoid
import tensorflow as tf

from utils.training_sdg_inner import train_o1, loss_o1, FAST_VERSION
#from utils.training_sdg_inner_reg import train_o1, loss_o1, FAST_VERSION
log.info("imported cython version: {}".format(FAST_VERSION))

class Node2Vec(object):
    def __init__(self, lr=0.2, workers=1, negative=0):

        self.workers = workers
        self.lr = float(lr)
        self.min_lr = 0.0001
        self.negative = negative
        self.window_size = 1
        self.MAX_EXP = 6

    def loss(self, model, edges):
        ret_loss = 0
        for edge in prepare_sentences(model, edges):
            assert len(edge) == 2, "edges have to be done by 2 nodes :{}".format(edge)
            edge_loss = np.log(
                sigmoid(np.dot(model.node_embedding[edge[1].index], model.node_embedding[edge[0].index].T)))
            assert edge_loss <= 0,"malformed loss"
            ret_loss -= edge_loss
        return ret_loss
    #
    # def loss(self, model, edges):
    #     loss = 0.0
    #     num_nodes = 0
    #
    #     for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), 250)):
    #         batch_loss = np.zeros(1, dtype=np.float32)
    #         batch_work = np.zeros(model.layer1_size, dtype=np.float32)
    #
    #
    #         batch_node = sum([loss_o1(model.node_embedding, edge, self.negative, model.table,
    #                              py_size=model.layer1_size, py_loss=batch_loss, py_work=batch_work) for edge in job if edge is not None])
    #         num_nodes += batch_node
    #         loss += batch_loss[0]
    #         # log.info("loss: {}\tnodes: {}".format(loss, num_nodes))
    #
    #     log.info(num_nodes)
    #     log.info(loss)
    #     return loss

    def train_ori(self, model, edges, G, chunksize=150, iter=1):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        assert model.node_embedding.dtype == np.float32

        log.info("O1 training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.corpus.shape[0] * edges.corpus.shape[1] * edges.n
        log.debug('total edges: %d' % total_node)
        start, next_report, node_count = time.time(), [5.0], [0]

        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()


        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            py_work = np.zeros(model.layer1_size, dtype=np.float32)

            while True:
                job = jobs.get(block=True)
                if job is None:  # data finished, exit
                    jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break

                lr = max(self.min_lr, self.lr * (1 - 1.0 * node_count[0]/total_node))
                #lr = self.lr 
                job_words = 0
                #out_i = 0
                for edge in job:
                    if edge is not None:
                        weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                        neg_l = []
                        #负样本node选取和主node不连通的点
                        min_node0, min_conn_tup = sorted(model.connected_path[model.vocab_t[edge[0].index]].items(), key=lambda x:x[1][0])[0]
                        min_conn0 = min_conn_tup[0]
                        min_node1, min_conn_tup = sorted(model.connected_path[model.vocab_t[edge[1].index]].items(), key=lambda x:x[1][0])[0]
                        min_conn1 = min_conn_tup[0]
                        for i in range(self.negative):
                            nodeidx = model.table[np.random.randint(model.table_size - 1)]
                            if (model.vocab_t[nodeidx] not in model.connected_path[model.vocab_t[edge[0].index]]
                                or (model.connected_path[model.vocab_t[edge[0].index]][model.vocab_t[nodeidx]][0] <= max(0.1,min_conn0))) \
                                and (model.vocab_t[nodeidx] not in model.connected_path[model.vocab_t[edge[1].index]]
                                or (model.connected_path[model.vocab_t[edge[1].index]][model.vocab_t[nodeidx]][1] <= max(0.1,min_conn1))):
                                neg_l.append(nodeidx)
                        if len(neg_l) == 0:
                            neg_l.append(model.vocab[min_node0].index)
                            neg_l.append(model.vocab[min_node1].index)
                        neg_np = np.asarray(neg_l)
                        if weight >= 0.0:
                            #job_words += sum(train_o1(model.node_embedding, edge, weight, lr, self.negative, model.table,
                            job_words += sum(train_o1(model.node_embedding, edge, lr, self.negative, neg_np,
                                         py_size=model.layer1_size, py_work=py_work) 
                                         for i in range(1))
                                         #for i in range(int(10 * weight)))
                #job_words = sum(train_o1(model.node_embedding, edge, lr, self.negative, model.table,
                #                         py_size=model.layer1_size, py_work=py_work) for edge in job if edge is not None)
                jobs.task_done()
                lock.acquire(timeout=30)
                try:
                    node_count[0] += job_words

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        log.info("PROGRESS: at %.2f%% \tnode_computed %d\talpha %.05f\t %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_node, node_count[0], lr, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 5.0  # don't flood the log, wait at least a second between progress reports
                finally:
                    lock.release()
        
        workers = [threading.Thread(target=worker_train, name='thread_'+str(i)) for i in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), chunksize)):
            jobs.put(job)

        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        log.info("training on %i words took %.1fs, %.0f words/s" %
                    (node_count[0], elapsed, node_count[0]/ elapsed if elapsed else 0.0))

    def train(self, model, edges, G, cluster_negtivate=False, nodeid2cluster={}, chunksize=150, iter=1):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        assert model.node_embedding.dtype == np.float32

        log.info("O1 training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.corpus.shape[0] * edges.corpus.shape[1] * edges.n
        log.debug('total edges: %d' % total_node)
        start, next_report, node_count = time.time(), [5.0], [0]

        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()

        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            py_work = np.zeros(model.layer1_size, dtype=np.float32)

            while True:
                job = jobs.get(block=True)
                if job is None:  # data finished, exit
                    jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break

                #lr = max(self.min_lr, self.lr * (1 - 1.0 * node_count[0]/total_node))
                lr = self.lr 
                job_words = 0
                for edge in job:
                    if edge is not None:
                        if cluster_negtivate:
                            node_set = set()
                            if model.vocab_t[edge[0].index] not in nodeid2cluster:
                                cls1 = -1
                            else:
                                cls1 = nodeid2cluster[model.vocab_t[edge[0].index]]
                                node_set.add(cls1)
                            if model.vocab_t[edge[1].index] not in nodeid2cluster:
                                cls2 = -1
                            else:
                                cls2 = nodeid2cluster[model.vocab_t[edge[1].index]]
                                node_set.add(cls2)
                            neg_l = []
                            #选择的负样本的node必须是有明确类别归属的
                            for i in range(self.negative):
                                nodeidx = model.table[np.random.randint(model.table_size)]
                                if model.vocab_t[nodeidx] not in nodeid2cluster:
                                    i-=1
                                    continue
                                else:
                                    cls_n = nodeid2cluster[model.vocab_t[nodeidx]]
                                #加入不同边限制 G 里存放的是nodeid，不是idx
                                if cls_n not in node_set and model.vocab_t[nodeidx] not in G[model.vocab_t[edge[0].index]]  \
                                        and model.vocab_t[nodeidx] not in G[model.vocab_t[edge[1].index]]:
                                    neg_l.append(nodeidx)
                            neg_np = np.asarray(neg_l)
                            weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                            if weight > 0.0 and len(neg_np) > 0:
                                #job_words += sum(train_o1(model.node_embedding, edge, lr, int(10 * (weight)) * self.negative, neg_np,
                                #            py_size=model.layer1_size, py_work=py_work) 
                                #            for i in range(1))
                                job_words += sum(train_o1(model.node_embedding, edge, weight, lr, self.negative, neg_np,
                                                py_size=model.layer1_size, py_work=py_work) 
                                                for i in range(1))
                            elif len(neg_np) == 0:
                                #job_words += sum(train_o1(model.node_embedding, edge, lr, 0, neg_np,
                                #            py_size=model.layer1_size, py_work=py_work) 
                                #            for i in range(1))
                                job_words += sum(train_o1(model.node_embedding, edge, weight, lr, 0, neg_np,
                                                py_size=model.layer1_size, py_work=py_work) 
                                                for i in range(1))
                        else:
                            weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                            if weight >= 0.1:
                                #job_words += sum(train_o1(model.node_embedding, edge, lr, int(10 * (weight)) * self.negative, model.table,
                                #            py_size=model.layer1_size, py_work=py_work) 
                                #            for i in range(1))
                                job_words += sum(train_o1(model.node_embedding, edge, weight, lr, self.negative, model.table,
                                         py_size=model.layer1_size, py_work=py_work) 
                                         for i in range(1))
                                         #for i in range(int(10 * weight)))
                #job_words = sum(train_o1(model.node_embedding, edge, lr, self.negative, model.table,
                #                         py_size=model.layer1_size, py_work=py_work) for edge in job if edge is not None)
                jobs.task_done()
                lock.acquire(timeout=30)
                try:
                    node_count[0] += job_words

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        log.info("PROGRESS: at %.2f%% \tnode_computed %d\talpha %.05f\t %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_node, node_count[0], lr, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 5.0  # don't flood the log, wait at least a second between progress reports
                finally:
                    lock.release()


        workers = [threading.Thread(target=worker_train, name='thread_'+str(i)) for i in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), chunksize)):
            jobs.put(job)

        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        log.info("training on %i words took %.1fs, %.0f words/s" %
                    (node_count[0], elapsed, node_count[0]/ elapsed if elapsed else 0.0))
    def train_reg(self, sess, pre, model, edges, G, cluster_negtivate=False, nodeid2cluster={}, iter=1, chunksize=150):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        assert model.node_embedding.dtype == np.float32

        log.info("O1 training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.corpus.shape[0] * edges.corpus.shape[1] * edges.n
        log.debug('total edges: %d' % total_node)
        start, next_report, node_count = time.time(), [5.0], [0]

        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()


        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            py_work = np.zeros(model.layer1_size, dtype=np.float32)

            while True:
                job = jobs.get(block=True)
                if job is None:  # data finished, exit
                    jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break

                lr = max(self.min_lr, self.lr * (1 - 1.0 * node_count[0]/total_node))
                #lr = self.lr 
                job_words = 0
                #pre=self.build_model(len(model.vocab), model.layer1_size, lamda = 0.0, learning_rate=lr)
                for edge in job:
                    if edge is not None:
                        x = []
                        y = []
                        x.append([edge[0].index, edge[1].index])
                        weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                        #y.append(weight)
                        y.append(1.0)
                        #for i in range(int(10 * (weight)) * self.negative):
                        for i in range(self.negative):
                            nodeidx = model.table[np.random.randint(model.table_size)]
                            if nodeidx != edge[0].index:
                                x.append([edge[0].index, nodeidx])
                                y.append(0.0)
                        feed_dict = {
                            pre.x: x,
                            pre.y: y,
                            pre.node_embeddings_init: model.node_embedding
                        }
                        #saver = tf.train.Saver()
                        _, loss, node_embeddings = sess.run([pre.d_updates, pre.reg_loss, pre.node_embeddings_n1],
                            feed_dict=feed_dict)
                        model.node_embedding[edge[0].index] = node_embeddings[edge[0].index]
                        x = []
                        y = []
                        x.append([edge[1].index, edge[0].index])
                        weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                        #y.append(weight)
                        y.append(1.0)
                        for i in range(self.negative):
                            nodeidx = model.table[np.random.randint(model.table_size)]
                            if edge[1].index != nodeidx:
                                x.append([edge[1].index, nodeidx])
                                y.append(0.0)
                        feed_dict = {
                            pre.x: x,
                            pre.y: y,
                            pre.node_embeddings_init: model.node_embedding
                        }
    
                        #saver = tf.train.Saver()
                        _, loss, node_embeddings = sess.run([pre.d_updates, pre.reg_loss, pre.node_embeddings_n1],
                            feed_dict=feed_dict)

                        #model.node_embedding = node_embeddings
                        model.node_embedding[edge[1].index] = node_embeddings[edge[1].index]
                        job_words += len(x)
                
                #log.info("train_loss: {}, node_embeddings = {}".format(loss, model.node_embedding))
            
                #saver.restore(sess, INNER_MODEL_FILE)
                #job_words = sum(train_o1(model.node_embedding, edge, lr, self.negative, model.table,
                #                         py_size=model.layer1_size, py_work=py_work) for edge in job if edge is not None)
                #job_words = len(x)
                jobs.task_done()
                lock.acquire(timeout=30)
                try:
                    node_count[0] += job_words

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        log.info("PROGRESS: at %.2f%% \tnode_computed %d\talpha %.05f\t %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_node, node_count[0], lr, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 5.0  # don't flood the log, wait at least a second between progress reports
                finally:
                    lock.release()


        workers = [threading.Thread(target=worker_train, name='thread_'+str(i)) for i in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()


        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), chunksize)):
            jobs.put(job)


        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()
        
        elapsed = time.time() - start
        log.info("training on %i words took %.1fs, %.0f words/s" %
                    (node_count[0], elapsed, node_count[0]/ elapsed if elapsed else 0.0))

    def build_model(self, node_size, emb_dims, lamda = 0.0, learning_rate=0.05):
        self.lamda = lamda  # regularization parameters
        self.learning_rate = learning_rate
        self.d_params = []
        self.node_size=node_size
        self.emb_dims=emb_dims
        self.initdelta = 0.05
        with tf.variable_scope('discriminator', reuse=None) as scope:
            # placeholder definition
            
            #self.node_embeddings_n1 = tf.get_variable("node_embeddings", [self.node_size, self.emb_dims], dtype=tf.float32,
            #            initializer=tf.random_normal_initializer(-self.initdelta, self.initdelta))
            self.node_embeddings_n1 = tf.Variable(
                        tf.random_uniform([self.node_size, self.emb_dims], 
                            minval=-self.initdelta, maxval=self.initdelta,
                            dtype=tf.float32))
            self.node_embeddings_n2 = tf.Variable(
                        tf.random_uniform([self.node_size, self.emb_dims], 
                            minval=-self.initdelta, maxval=self.initdelta,
                            dtype=tf.float32))
                            
        self.x = tf.placeholder(tf.float32,  [None, 2])
        self.y = tf.placeholder(tf.float32,  [None])
        self.node_embeddings_init = tf.placeholder(tf.float32,  [self.node_size, self.emb_dims])
        tf.assign(self.node_embeddings_n1, self.node_embeddings_init)
        tf.assign(self.node_embeddings_n2, self.node_embeddings_init)
        #self.is_train = tf.placeholder(tf.int32)

        #trans to int
        node1, node2 = tf.split(self.x, [1, 1], 1)
        node1_int = tf.cast(node1, tf.int32)
        node2_int = tf.cast(node2, tf.int32)
        
        self.n1_embedding = tf.nn.embedding_lookup(self.node_embeddings_n1, node1_int)

        self.n2_embedding = tf.nn.embedding_lookup(self.node_embeddings_n2, node2_int)
        
        #
        self.n1_embedding_s = tf.squeeze(self.n1_embedding, axis=1)
        self.n2_embedding_s = tf.squeeze(self.n2_embedding, axis=1)
        #计算cosine
        #n1_deno = tf.sqrt(tf.reduce_sum(self.n1_embedding * self.n1_embedding, 1))
        #n2_deno = tf.sqrt(tf.reduce_sum(self.n2_embedding * self.n2_embedding, 1))
        #nu = tf.reduce_sum(self.n1_embedding * self.n2_embedding, 1)
        #self.mut = tf.div(nu, n1_deno * n2_deno +1e-8)

        #计算dot
        self.mut = tf.sigmoid(tf.clip_by_value(
                    tf.reduce_sum(self.n1_embedding_s * self.n2_embedding_s, 1), 
                    -self.MAX_EXP, 
                    self.MAX_EXP))
        
        self.d_params = [self.node_embeddings_n1]
        self.reg_loss = 0.1 * tf.reduce_mean((self.mut - self.y) * (self.mut - self.y))
        #self.loss = tf.reduce_mean(self.weight * tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, \
        #        logits=self.logits) \
        #        + self.lamda * (tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        #self.d_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.reg_loss, var_list=self.d_params)
        #self.d_updates = d_opt.minimize(self.reg_loss)
        
        #all_rating for output
        self.all_rating = self.mut

    def train_reg_struct(self, sess, pre, model, edges, G, cluster_negtivate=False, nodeid2cluster={}, iter=1, chunksize=150):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        assert model.node_embedding.dtype == np.float32

        log.info("O1 training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        edges = RepeatCorpusNTimes(edges, iter)
        total_node = edges.corpus.shape[0] * edges.corpus.shape[1] * edges.n
        log.debug('total edges: %d' % total_node)
        start, next_report, node_count = time.time(), [5.0], [0]

        #mean field
        #print("model.node_embedding = ", model.node_embedding)
        #print("model.w2.T = ", model.w2.T)
        node_emb_tmp = np.zeros((model.vocab_size, model.layer1_size), dtype=np.float32)
        loop = 100
        log.info("i = 0, model.node_embedding = {}".format(model.node_embedding[1]))
        for i in range(loop):
            for node in G.nodes():
                    #node_emb = tf.nn.embedding_lookup(self.node_embeddings, model.vocab[node])
                #tmp = np.zeros(model.layer1_size, dtype=np.float32)
                #for nnodeid in G[node]:
                #    nodeidx = model.vocab[nnodeid].index
                #    tmp = tmp + model.node_embedding[nodeidx]
                tmp = np.sum([G[node][nodeid]['weight'] * 
                    model.node_embedding[model.vocab[nodeid].index] 
                                for nodeid in G[node]], axis=0)
                    #print("near nodeidx = ", nodeidx, ", emb = ", model.node_embedding[nodeidx])
                #print("np.matmul(tmp, model.w2.T) = ", np.matmul(tmp, model.w2.T))
                #model.node_embedding[model.vocab[node].index] = np.maximum(0, np.matmul(tmp, model.w2.T))
                #print("ori nodeidx = ", model.vocab[node].index, ", emb = ", model.node_embedding[model.vocab[node].index])
                #node_emb_tmp[model.vocab[node].index] = 1 / (1 + np.exp(-np.matmul(tmp, model.w2.T)))
                node_emb_tmp[model.vocab[node].index] = 10 * np.tanh(np.matmul(tmp, model.w2.T))
                #node_emb_tmp[model.vocab[node].index] = np.exp(np.matmul(tmp, model.w2.T))
                #node_emb_tmp[model.vocab[node].index] = np.matmul(tmp, model.w2.T)
                #e_x = np.exp(np.matmul(tmp, model.w2.T) - np.max(np.matmul(tmp, model.w2.T)))
                #node_emb_tmp[model.vocab[node].index] = e_x / e_x.sum()
            #print("node_emb_tmp = ", node_emb_tmp)
            #print("model.node_embedding = ", model.node_embedding)
            model.node_embedding = self.normal(node_emb_tmp.copy())
            if i == loop - 1 or i == loop - 2:
                log.info("i = {}, node_emb_tmp = {}".format(i, node_emb_tmp[1]))
            #    print("i = ", i,", model.node_embedding = ", model.node_embedding)

        #int(sum(v.count * v.sample_probability for v in self.vocab.values()))
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        train_jobs = Queue(maxsize=2*self.workers)
        lock = threading.Lock()
        

        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            #py_work = np.zeros(model.layer1_size, dtype=np.float32)

            while True:
                job = jobs.get(block=True)
                if job is None:  # data finished, exit
                    jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break
                
                #log.info("thread start!")
                #lr = max(self.min_lr, self.lr * (1 - 1.0 * node_count[0]/total_node))
                lr = self.lr 
                job_words = 0
                #pre=self.build_model(len(model.vocab), model.layer1_size, lamda = 0.0, learning_rate=lr)
                x1 = []
                x2 = []
                y = []
                #cur = 0
                for edge in job:
                    if edge is not None:
                        #cur+=1
                        #if cur % 100 == 0:
                        #    log.info("edge[0].index = {}".format(edge[0].index))
                        #x.append([edge[0].index, edge[1].index)
                        edge_0_emb = np.sum([G[model.vocab_t[edge[0].index]][nodeid]['weight'] *
                            model.node_embedding[model.vocab[nodeid].index] 
                                        for nodeid in G[model.vocab_t[edge[0].index]]], axis=0)
                        #edge_0_emb = model.node_embedding[edge[0].index] 
                        edge_1_emb = np.sum([G[model.vocab_t[edge[1].index]][nodeid]['weight'] *
                            model.node_embedding[model.vocab[nodeid].index] 
                                        for nodeid in G[model.vocab_t[edge[1].index]]], axis=0)
                        #edge_1_emb = model.node_embedding[edge[1].index]
                        x1.append(edge_0_emb)
                        #print("edge[0].index = ", edge[0].index)
                        #print("0 nebor g = ", [nodeid for nodeid in G[model.vocab_t[edge[0].index]]])
                        #print("edge[1].index = ", edge[1].index)
                        #print("1 nebor g = ", [nodeid for nodeid in G[model.vocab_t[edge[1].index]]])
                        #print("model.vocab_t = ", model.vocab_t)
                        x2.append(edge_1_emb)
                        
                        #print("1 nebor g = ", G[model.vocab_t[edge[1].index]])
                        weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                        #y.append(weight)
                        y.append(1.0)
                        #for i in range(int(10 * (weight)) * self.negative):
                        for i in range(self.negative):
                            nodeidx = model.table[np.random.randint(model.table_size - 1)]
                            if nodeidx != edge[0].index and \
                            (model.vocab_t[nodeidx] not in model.connected_path[model.vocab_t[edge[0].index]]
                            or (model.connected_path[model.vocab_t[edge[0].index]][model.vocab_t[nodeidx]][0] < 0.1)):
                                x1.append(edge_0_emb)
                                x2.append(np.sum([G[model.vocab_t[nodeidx]][nodeid]['weight'] *
                                    model.node_embedding[model.vocab[nodeid].index] 
                                        for nodeid in G[model.vocab_t[nodeidx]]], axis=0)
                                )
                                #x2.append(model.node_embedding[nodeidx]) 
                                y.append(0.0)
                            else:
                                i -= 1
                            if nodeidx != edge[1].index and \
                            (model.vocab_t[nodeidx] not in model.connected_path[model.vocab_t[edge[1].index]]
                            or (model.connected_path[model.vocab_t[edge[1].index]][model.vocab_t[nodeidx]][0] < 0.1)):
                                x1.append(edge_1_emb)
                                x2.append(np.sum([G[model.vocab_t[nodeidx]][nodeid]['weight'] *
                                    model.node_embedding[model.vocab[nodeid].index] 
                                        for nodeid in G[model.vocab_t[nodeidx]]], axis=0)
                                )
                                #x2.append(model.node_embedding[nodeidx]) 
                                y.append(0.0)
                            else:
                                i-=1
                #log.info("edge end!")
                #print("model.node_embedding = ", model.node_embedding)
                
                #for i in range(1):
                #    feed_dict = {
                #        pre.x1: x1,
                #        pre.x2: x2,
                #        pre.y: y,
                #        pre.w2_init: model.w2
                #    }
                            #saver = tf.train.Saver()
                    #print("model.w2 = ", model.w2)
                            #_, loss, node_embeddings, w2 = sess.run([pre.d_updates, pre.reg_loss, pre.node_embeddings, pre.w2],
                    #_, loss, w2, mut, mut_ori = sess.run([pre.d_updates, pre.reg_loss, pre.w2, pre.mut, pre.mut_ori],
                    #        feed_dict=feed_dict)
                    #log.info("iter = {}, loss = {}".format(i, loss))
                    #if i == loop - 1:
                    #    print("y = ", y)
                    #    print("mut = ", mut)
                    #    print("w2.T = ", w2.T)
                #print("mut_ori = ", mut_ori)
                #loop = 10
                #for i in range(loop):
                #    for node in G.nodes():
                #            #node_emb = tf.nn.embedding_lookup(self.node_embeddings, model.vocab[node])
                #        tmp = np.zeros(model.layer1_size, dtype=np.float32)
                #        for nnodeid in G[node]:
                #            nodeidx = model.vocab[nnodeid].index
                #            #print("nodeidx = ", nodeidx)
                #            tmp = tmp + model.node_embedding[nodeidx]
                #        #model.node_embedding[model.vocab[node].index] = np.maximum(0, np.matmul(tmp, w2.T))
                #        #model.node_embedding[model.vocab[node].index] = np.exp(np.matmul(tmp, w2.T))
                #        model.node_embedding[model.vocab[node].index] = np.matmul(tmp, w2.T)
                #        #e_x = np.exp(np.matmul(tmp, w2.T) - np.max(np.matmul(tmp, w2.T)))
                #        #model.node_embedding[model.vocab[node].index] = e_x / e_x.sum()
                #        #model.node_embedding = node_embeddings
                #print("model.node_embedding_next = ", model.node_embedding)
                    #model.w2 = w2
                    #print("model.w2_next = ", model.w2)
                        #x = []
                        #y = []
                        #x.append([edge[1].index, edge[0].index])
                        #weight = G[model.vocab_t[edge[0].index]][model.vocab_t[edge[1].index]]['weight']
                        #y.append(weight)
                        #y.append(1.0)
                        #for i in range(self.negative):
                        #    nodeidx = model.table[np.random.randint(model.table_size)]
                        #    if edge[1].index != nodeidx:
                        #        x.append([edge[1].index, nodeidx])
                        #        y.append(0.0)
                        #feed_dict = {
                        #    pre.x: x,
                        #    pre.y: y,
                        #    pre.node_embeddings_init: model.node_embedding
                        #}
    
                        #saver = tf.train.Saver()
                        #_, loss, node_embeddings = sess.run([pre.d_updates, pre.reg_loss, pre.node_embeddings_n1],
                        #    feed_dict=feed_dict)

                        #model.node_embedding = node_embeddings
                        #model.node_embedding[edge[1].index] = node_embeddings[edge[1].index]
                #log.info("thread end!")
                job_words += len(y)
                
                #log.info("train_loss: {}, node_embeddings = {}".format(loss, model.node_embedding))
            
                #saver.restore(sess, INNER_MODEL_FILE)
                #job_words = sum(train_o1(model.node_embedding, edge, lr, self.negative, model.table,
                #                         py_size=model.layer1_size, py_work=py_work) for edge in job if edge is not None)
                #job_words = len(x)
                #log.info("train_jobs put!")
                #log.info("train_jobs_full = {}".format(train_jobs.full()))
                train_jobs.put([x1, x2, y])
                #train_jobs.put([x1, x2, y], block=False)
                #log.info("train_jobs put end!")
                jobs.task_done()
                lock.acquire(timeout=30)
                try:
                    node_count[0] += job_words

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        log.info("PROGRESS: at %.2f%% \tnode_computed %d\talpha %.05f\t %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_node, node_count[0], lr, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 5.0  # don't flood the log, wait at least a second between progress reports
                        #log.info("jobs.qsize() = {}, train_jobs.qsize() = {}".format(jobs.qsize(), train_jobs.qsize()))
                    #train_jobs.put([x1, x2, y])
                finally:
                    lock.release()
        
        def worker_train_tf():
            while True:
                #log.info(" start1, train_jobs = {}".format(train_jobs.qsize()))
                job = train_jobs.get(block=True)
                if job is None:  # data finished, exit
                    train_jobs.task_done()
                    # print('thread %s break' % threading.current_thread().name)
                    break
                #log.info(" start2, train_jobs = {}".format(train_jobs.qsize()))
                x1, x2, y = job
                for i in range(1):
                    feed_dict = {
                        pre.x1: x1,
                        pre.x2: x2,
                        pre.y: y,
                        pre.w2_init: model.w2
                    }            
                    #_, loss, w2, mut, mut_ori = sess.run([pre.d_updates, pre.reg_loss, pre.w2, pre.mut, pre.mut_ori],
                    #_, loss, w2, lr = sess.run([pre.d_updates, pre.reg_loss, pre.w2, pre.learning_rate],
                    _, loss, w2 = sess.run([pre.d_updates, pre.reg_loss, pre.w2],
                                feed_dict=feed_dict)
                    log.info("iter = {}, loss = {}".format(i, loss))
                    model.w2 = w2
                train_jobs.task_done()
                #log.info("train_jobs.qsize() = {}".format(train_jobs.qsize()))


        workers = [threading.Thread(target=worker_train, name='thread_'+str(i)) for i in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()
            #log.info("thread = {} start!".format(thread))
        
        train_worker = [threading.Thread(target=worker_train_tf, name='train_thread_'+str(i)) for i in range(1)]
        for thread in train_worker:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()
            #log.info("train thread = {} start!".format(thread))

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, edges), chunksize)):
            jobs.put(job)
            #log.info("jobs.qsize() = {}".format(jobs.qsize()))

        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!
        
        for _ in range(1):
            train_jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()
        
        for thread in train_worker:
            thread.join()
        
        elapsed = time.time() - start
        log.info("training on %i words took %.1fs, %.0f words/s" %
                    (node_count[0], elapsed, node_count[0]/ elapsed if elapsed else 0.0))
    
    def build_model_struct(self, node_size, emb_dims, G, model, lamda = 0.0, learning_rate=0.05, loop=1):
        self.lamda = lamda  # regularization parameters
        self.learning_rate = learning_rate
        self.d_params = []
        self.node_size=node_size
        self.emb_dims=emb_dims
        self.initdelta = 0.05
        with tf.variable_scope('discriminator', reuse=None) as scope:
            # placeholder definition
            
            #self.node_embeddings_n1 = tf.get_variable("node_embeddings", [self.node_size, self.emb_dims], dtype=tf.float32,
            #            initializer=tf.random_normal_initializer(-self.initdelta, self.initdelta))
            #self.node_embeddings_n1 = tf.Variable(
            #            tf.random_uniform([self.node_size, self.emb_dims], 
            #                minval=-self.initdelta, maxval=self.initdelta,
            #                dtype=tf.float32))
            self.w2 = tf.Variable(
                        tf.random_uniform([self.emb_dims, self.emb_dims], 
                            minval=-self.initdelta, maxval=self.initdelta,
                            dtype=tf.float32))
            #self.node_embeddings = tf.Variable(
            #            tf.random_uniform([self.node_size, self.emb_dims], 
            #                minval=-self.initdelta, maxval=self.initdelta,
            #                dtype=tf.float32)
            #)
            #self.node_embeddings_next = tf.Variable(
            #            tf.random_uniform([self.node_size, self.emb_dims], 
            #                minval=-self.initdelta, maxval=self.initdelta,
            #                dtype=tf.float32)
            #)
                            
        self.x1 = tf.placeholder(tf.float32,  [None, self.emb_dims])
        self.x2 = tf.placeholder(tf.float32,  [None, self.emb_dims])
        self.y = tf.placeholder(tf.float32,  [None])
        self.w2_init = tf.placeholder(tf.float32,  [self.emb_dims, self.emb_dims])
        #self.current_epoch = tf.Variable(0)
        #self.node_embeddings_init = tf.placeholder(tf.float32,  [self.node_size, self.emb_dims])
        #tf.assign(self.node_embeddings, self.node_embeddings_init)
        tf.assign(self.w2, self.w2_init)
        #tf.assign(self.node_embeddings_n2, self.node_embeddings_init)
        #self.is_train = tf.placeholder(tf.int32)

        #loop all node product embeds
        #one_tensor = tf.ones([1], tf.int32)
        #for i in range(loop):
        #    for node in G.nodes():
        #        #node_emb = tf.nn.embedding_lookup(self.node_embeddings, model.vocab[node])
        #        tmp = tf.zeros([self.emb_dims])
        #        for nnodeid in G[node]:
        #            nodeidx = model.vocab[nnodeid].index
        #            #print("nodeidx = ", nodeidx)
        #            nnode_embedding = self.node_embeddings[nodeidx]
        #            #print("nnode = ", nnode_embedding)
        #            tmp = tmp + nnode_embedding
        #            #print("tmp = ", tmp)
        #        tmp = tf.reshape(tmp, [1,self.emb_dims])
        #        #print("indesc = ", one_tensor * model.vocab[node].index)
        #        #print("w2*tmp", tf.matmul(self.w2, tmp, transpose_b=True))
        #        self.set_value(self.node_embeddings, one_tensor * model.vocab[node].index, 
        #                tf.reshape(tf.nn.relu(tf.matmul(self.w2, tmp, transpose_b=True)), [1, self.emb_dims]))

        #trans to int
        #node1, node2 = tf.split(self.x, [1, 1], 1)
        #node1_int = tf.cast(node1, tf.int32)
        #node2_int = tf.cast(node2, tf.int32)
        
        #self.n1_embedding = tf.nn.embedding_lookup(self.node_embeddings, node1_int)

        #self.n2_embedding = tf.nn.embedding_lookup(self.node_embeddings, node2_int)
        
        #
        #self.n1_embedding_s = tf.squeeze(self.n1_embedding, axis=1)
        #self.n2_embedding_s = tf.squeeze(self.n2_embedding, axis=1)
        #self.n1_embedding_s = tf.nn.sigmoid(tf.matmul(self.x1, self.w2, transpose_b=True))
        #self.n2_embedding_s = tf.nn.sigmoid(tf.matmul(self.x2, self.w2, transpose_b=True))
        #self.n1_embedding_s = tf.nn.softmax(tf.matmul(self.x1, self.w2, transpose_b=True))
        #self.n2_embedding_s = tf.nn.softmax(tf.matmul(self.x2, self.w2, transpose_b=True))
        #self.n1_embedding_s = tf.nn.relu(tf.matmul(self.x1, self.w2, transpose_b=True))
        #self.n2_embedding_s = tf.nn.relu(tf.matmul(self.x2, self.w2, transpose_b=True))
        self.n1_embedding_s = 10 * tf.nn.tanh(tf.matmul(self.x1, self.w2, transpose_b=True))
        self.n2_embedding_s = 10 * tf.nn.tanh(tf.matmul(self.x2, self.w2, transpose_b=True))
        #self.n1_embedding_s = tf.matmul(self.x1, self.w2, transpose_b=True)
        #self.n2_embedding_s = tf.matmul(self.x2, self.w2, transpose_b=True)
        #self.n1_embedding_s = tf.matmul(self.x1, self.w2, transpose_b=True)
        #self.n2_embedding_s = tf.matmul(self.x2, self.w2, transpose_b=True)
        #计算cosine
        #n1_deno = tf.sqrt(tf.reduce_sum(self.n1_embedding * self.n1_embedding, 1))
        #n2_deno = tf.sqrt(tf.reduce_sum(self.n2_embedding * self.n2_embedding, 1))
        #nu = tf.reduce_sum(self.n1_embedding * self.n2_embedding, 1)
        #self.mut = tf.div(nu, n1_deno * n2_deno +1e-8)

        #print("self.n1_embedding_s = ", self.n1_embedding_s)
        #计算dot
        self.mut_ori = tf.reduce_sum(self.n1_embedding_s * self.n2_embedding_s, 1)
        self.mut = tf.sigmoid(tf.clip_by_value(
                    tf.reduce_sum(self.n1_embedding_s * self.n2_embedding_s, 1), 
                    -self.MAX_EXP, 
                    self.MAX_EXP))
        
        self.d_params = [self.w2]
        self.reg_loss = tf.reduce_mean((self.mut - self.y) * (self.mut - self.y))
        #self.loss = tf.reduce_mean(self.weight * tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, \
        #        logits=self.logits) \
        #        + self.lamda * (tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2)

        #d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        #self.learning_rate = tf.train.exponential_decay(0.025,  
        #                                   self.current_epoch,  
        #                                   decay_steps=100,  
        #                                   decay_rate=0.96)
        d_opt = tf.train.AdamOptimizer(self.learning_rate)
        #self.d_updates = d_opt.minimize(self.reg_loss, var_list=self.d_params, global_step=self.current_epoch)
        self.d_updates = d_opt.minimize(self.reg_loss, var_list=self.d_params)
        #self.d_updates = d_opt.minimize(self.reg_loss)
        
        #all_rating for output
        self.all_rating = self.mut
    
    def normal(self, matrix):
        #dis = np.max(matrix) - np.min(matrix)
        #return 10 * (2 * (matrix - np.min(matrix) - dis/2 ) / dis)
        return matrix / 10 ** int(np.log10(np.abs(np.min(matrix))))
    
    def mean_field(self, G, model, nodeidx):
        #one_tensor = tf.ones([1], tf.int32)
        tmp = tf.zeros([self.emb_dims])
        for nnodeid in G[nodeidx]:
            nodeidx = model.vocab[nnodeid].index
            #print("nodeidx = ", nodeidx)
            tmp += self.node_embeddings[nodeidx]
            #print("nnode = ", nnode_embedding)
            tmp = tmp + nnode_embedding
        return tmp
        #        tmp = tf.reshape(tmp, [1,self.emb_dims])
    
    def set_value(self, matrix, x, val):
        # 提取出要更新的行
        #row = tf.gather(matrix, x)
        # 构造这行的新数据
        #new_row = val
        # 使用 tf.scatter_update 方法进正行替换
        matrix.assign(tf.scatter_update(matrix, x, val))
    
    def set_value_fast(self, matrix, x, y, val):
        # 得到张量的宽和高，即第一维和第二维的Size
        w = int(matrix.get_shape()[0])
        h = int(matrix.get_shape()[1])
        # 构造一个只有目标位置有值的稀疏矩阵，其值为目标值于原始值的差
        val_diff = val - matrix[x][y]
        diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[x, y], values=[val_diff], dense_shape=[w, h]))
        # 用 Variable.assign_add 将两个矩阵相加
        matrix.assign_add(diff_matrix)
