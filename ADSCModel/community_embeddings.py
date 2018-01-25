__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
from utils.embedding import chunkize_serial

from scipy.stats import multivariate_normal
import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


class Community2Vec(object):
    '''
    Class that train the community embedding
    '''
    def __init__(self, lr):
        self.lr = lr

    def fit(self, model, reg_covar=0, n_init=10, means_init=None):
        '''
        Fit the GMM model with the current node embedding and save the result in the model
        :param model: model injected to add the mixture parameters
        '''
        if means_init is None:
            self.g_mixture = mixture.GaussianMixture(n_components=model.k,
                                                 reg_covar=reg_covar,
                                                 #covariance_type='full',
                                                 #covariance_type='tied',
                                                 covariance_type='diag',
                                                 #covariance_type='spherical',
                                                 n_init=n_init,
                                                 max_iter=100,
                                                 warm_start=True)
        else:
            self.g_mixture = mixture.GaussianMixture(n_components=model.k,
                                                 reg_covar=reg_covar,
                                                 #covariance_type='full',
                                                 #covariance_type='tied',
                                                 covariance_type='diag',
                                                 #covariance_type='spherical',
                                                 n_init=n_init,
                                                 means_init=means_init,
                                                 max_iter=100,
                                                 warm_start=True)

        log.info("Fitting: {} communities".format(model.k))
        for i in range(1):
            model.random_choice(20000)
            self.g_mixture.fit(model.random_node_embedding)
        #self.g_mixture.fit(model.node_embedding)
        log.info("Fitting: {} communities final！".format(model.k))

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = self.g_mixture.means_.astype(np.float32)
        model.covariance_mat = self.g_mixture.covariances_.astype(np.float32)
        model.inv_covariance_mat = self.g_mixture.precisions_.astype(np.float32)
        model.pi = self.g_mixture.predict_proba(model.node_embedding).astype(np.float32)
        #在这里可以对pi做正则

        # model.c = self.g_mixture.degrees_of_freedom_.astype(np.float32)
        # model.B = self.g_mixture.covariance_prior_.astype(np.float32)

    def pdf_multivariate_gauss(self, x, mu, cov):
        '''
        Caculate the multivariate normal density (pdf)

        Keyword arguments:
            x = numpy array of a "d x 1" sample vector
            mu = numpy array of a "d x 1" mean vector
            cov = "numpy array of a d x d" covariance matrix
        '''
        assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
        assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
        assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
        assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
        assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
        part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
        part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
        #log.info('part1 = {}, part2 = {}, np.linalg.det(cov) = {}, np.linalg.inv(cov) = {}'.format(
        #            part1, part2, np.linalg.det(cov), np.linalg.inv(cov)))
        return float(part1 * np.exp(part2))
    
    def predict(self, nodes, model):
        i=0
        j=0
        K=3
        #log.info('predict = {}'.format(self.g_mixture.predict(model.node_embedding).astype(np.float32)[:3]))
        model.probability = np.zeros((model.vocab_size, model.k), dtype=np.float64)
        for com in range(model.k):
            #rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
            for node_index in range(model.vocab_size):
                input = model.node_embedding[node_index]
                # check if can be done as matrix operation
                pdf = multivariate_normal.pdf(input, mean=model.centroid[com], cov=model.covariance_mat[com], allow_singular=True)
                #pdf = multivariate_normal.pdf(input.reshape(-1,1), mean=model.centroid[com].reshape(-1,1), cov=model.covariance_mat[com])
                #pdf = self.pdf_multivariate_gauss(input.reshape(-1,1), mu=model.centroid[com].reshape(-1,1), cov=model.covariance_mat[com])
                #pdf = self.g_mixture.score_samples(input.reshape(1,-1))
                model.probability[node_index, com] = pdf * model.pi[node_index, com]
                #if i < 3:
                    #log.info(multivariate_normal.pdf(model.node_embedding, mean=model.centroid[com], cov=model.covariance_mat[com])[:3])
                    #log.info('value = {}'.format(multivariate_normal.pdf([-0.25331852, -0.00491518], 
                    #        mean = [-0.4938415,-0.00804068], cov = [[ 0.16878246,0.0025839 ],[ 0.0025839,0.00028859]])))
                #    log.info('input = {}, cen = {}, cov = {}, \
                #            pdf = {}, pi = {}, pro = {}'.format(
                #            input, model.centroid[com], 
                #            model.covariance_mat[com], pdf,
                #            model.pi[node_index, com],
                #            model.probability[node_index, com]))
                #    i+=1
            #if j < 3:
                #max_idx = np.argpartition(model.probility[node_index],-K)[-K:]
            #    log.info('max_com = {}'.format(model.probability[node_index]))
            #    j+=1
        for node_index in range(model.vocab_size):
            try:
                model.probability[node_index] /= sum(model.probability[node_index])
            except:
                log.info("node_index = {}, model.probability[node_index] = {}".format(node_index, model.probability[node_index]))
                continue
                

    def loss(self, nodes, model, beta, chunksize=150):
        """
        Forward function used to compute o3 loss
        :param input_labels: of the node present in the batch
        :param model: model containing all the shared data
        :param beta: trade off param
        """
        ret_loss = 0
        for node_index in chunkize_serial(map(lambda x: model.vocab[x].index, nodes), chunksize):
            input = model.node_embedding[node_index]

            batch_loss = np.zeros(len(node_index), dtype=np.float32)
            for com in range(model.k):
                rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
                # check if can be done as matrix operation
                #batch_loss += rd.logpdf(input).astype(np.float32) * model.pi[node_index, com]
                batch_loss += rd.logpdf(input).astype(np.float32) * model.pi[node_index, com]

            #ret_loss = abs(batch_loss.sum())
            ret_loss = -(batch_loss.sum())

        return ret_loss * (beta/model.k)

    def train(self, nodes, model, beta, chunksize=150, iter=1):
        for _ in range(iter):
            grad_input = np.zeros(model.node_embedding.shape).astype(np.float32)
            for node_index in chunkize_serial(map(lambda node: model.vocab[node].index,
                                                  filter(lambda node: node in model.vocab and (model.vocab[node].sample_probability >= 1.0 or model.vocab[node].sample_probability >= np.random.random_sample()), nodes)), chunksize):
                input = model.node_embedding[node_index]
                batch_grad_input = np.zeros(input.shape).astype(np.float32)

                for com in range(model.k):
                    diff = np.expand_dims(input - model.centroid[com], axis=-1)
                    m = model.pi[node_index, com].reshape(len(node_index), 1, 1) * (model.inv_covariance_mat[com])

                    batch_grad_input += np.squeeze(np.matmul(m, diff), axis=-1)
                grad_input[node_index] += batch_grad_input


            grad_input *= (beta/model.k)

            model.node_embedding -= (grad_input.clip(min=-0.25, max=0.25)) * self.lr
