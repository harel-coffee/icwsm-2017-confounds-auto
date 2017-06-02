from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from grid import Grid
from helpers import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import KFold
from sklearn.calibration import calibration_curve
from collections import defaultdict
from scipy.stats import pearsonr
from progressbar import ProgressBar
from datetime import datetime

import inspect
import numpy as np
import itertools
import json

class PreliminaryStudy:
    """
    Class that wraps the preliminary study used in adjusting for unobserved confounding variables.

    The preliminary study is trained to predict the value of a confounding variable z given a subset of all the features X and an unbiased dataset.
    At fitting time, it builds an unbiased dataset (when ``bias`` is set to 0.5) and select in ``X`` the ``max_features`` most predictive features of `z`. It then computes metrics using cross-validation and finally fits a logistic regression classifier on the whole subdataset.
    """
    def __init__(self, bias, size, noise=None, max_features=None, kfolds=3, metrics=None, metrics_thresh=None, epsilon=None, calibration_bins=10, rand=None):
        """
        :param bias: bias to introduce in the preliminary study (float between 0 and 1, 0.5 indicates no bias).
        :param size: size of the dataset to create for the preliminary study.
        :param noise: quantity of uniform noise to introduce in the data used to train the preliminary study (float between 0 and 1).
        :param max_features: maximum number of features to keep for the preliminary study.
        :param kfolds: number of folds in preliminary study cross-validation.
        :param metrics: list of metrics to compute given z_true and z_pred.
        :param calibration_bins: number of bins to use in order to build the calibration curve. 
        """
        self.bias              = bias
        self.calibration_bins  = calibration_bins
        self.calibration_curve = defaultdict(list)
        self.cm                = None
        self.features          = None
        self.kfolds            = kfolds
        self.max_features      = max_features
        self.metrics           = metrics
        self.metrics_thresh    = metrics_thresh
        self.epsilon           = epsilon
        self.noise             = noise
        self.scores            = defaultdict(list)
        self.scores_raw        = defaultdict(list)
        self.size              = size
        self.var_err           = None
        self.rand              = np.random.RandomState(123456) if rand is None else rand

    def __str__(self):
        return ",".join([k + "=" + str(self.__dict__[k]) for k in ['bias', 'size', 'noise', 'max_features']])

    def __repr__(self):
        return self.__str__()
    
    def _cv(self, X, z):
        """
        Runs cross validation on subdataset and computes metrics.
        """
        dim_z = len(set(z))
        len_x = X.shape[0]

        self.cm = np.zeros((dim_z, dim_z))
        folds = KFold(len_x, self.kfolds, shuffle=True, random_state=self.rand)

        # Compute aggregated confusion matrix
        for ifold, (tr_idx, te_idx) in enumerate(folds):
            X_tr, X_te = X[tr_idx], X[te_idx]
            z_tr, z_te = z[tr_idx], z[te_idx]
            
            if self.noise:
                z_tr = uniform_noise_1d(z_tr, self.noise, rand=self.rand)

            self.clf.fit(X_tr, z_tr)
            z_pred = self.clf.predict(X_te)
            z_prob = self.clf.predict_proba(X_te)
            z_prob_pos = z_prob[:, 1]
            
            # Compute metrics score
            if self.metrics is not None:
                for metric in self.metrics:
                    self.scores_raw[metric.__qualname__].append(metric(z_te, z_pred))
                    
            # Compute metrics that require the use of the threshold
            if self.metrics_thresh is not None:
                for metric_thresh in self.metrics_thresh:
                    self.scores_raw[metric_thresh.__qualname__].append(metric_thresh(z_te, z_prob, self.epsilon))
                    
            # Compute confusion matrix
            self.cm += confusion_matrix(z_te, z_pred)
            
            # Compute binned probabilities for calibration curve
            prob_true, prob_pred = calibration_curve(z_te, z_prob_pos, n_bins=self.calibration_bins)
            for pt, pp in zip(prob_true, prob_pred):
                self.calibration_curve[pp].append(pt)

        if self.metrics is not None:
            for metric in self.metrics:
            #for metric_name, scores in self.scores_raw.items():
                metric_name = metric.__qualname__
                scores = self.scores_raw[metric_name]
                self.scores[metric_name] = float(np.mean(scores))
                
        self.p_zpgz = self.cm / np.sum(self.cm,axis=1).reshape(-1,1)
        self.p_zgzp = (self.cm / np.sum(self.cm,axis=0)).T

    def fit(self, X, y, z):
        rows = make_confounding_data(y, z, self.bias, self.size, rand=self.rand)
        self.rows = rows
        X = X[rows]
        y = y[rows]
        z = z[rows]
        
        self.corr, self.pval = pearsonr(y, z)
        if self.max_features:
            kbest = SelectKBest(chi2, k=self.max_features).fit(X, z)
            self.features = kbest.get_support(indices=True)
            X = kbest.transform(X)
            
        self.clf = LogisticRegression(class_weight='balanced')

        # Compute p(z|z~) and p(z~|z) using cross-validation
        self._cv(X,z)

        # Fit on the whole dataset for later use
        if self.noise:
            z = uniform_noise_1d(z, self.noise, rand=self.rand)
        self.clf.fit(X, z)
        return self

    def predict(self, X, rm_feature_cols=False):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X, rm_feature_cols=False):
        if rm_feature_cols and self.features is not None:
            X = X[:, self.features]

        return self.clf.predict_proba(X)
    

class Benchmark():
    def __init__(self, X, y, z, grids_param):
        self.X = X
        self.y = y
        self.z = z
        self.xl, self.xd = X.shape
        self.grids_param = grids_param
        self.done = []
        self.metrics = [f1_score, precision_score, recall_score, accuracy_score]
        
    def run(self, output_file, ntrials=3, rand=np.random.RandomState(123456)):
        tr_arg_names, tr_arg_values = zip(*self.grids_param['train'].items())
        te_arg_names, te_arg_values = zip(*self.grids_param['test'].items())
        count = 0

        grids = dict(
            prestudy = Grid(self.grids_param['prestudy']),
            mainstudy = Grid(self.grids_param['mainstudy']),
            tr = [_ for _ in itertools.product(*tr_arg_values)],
            te = [_ for _ in itertools.product(*te_arg_values)]
        )
        total_models = np.product([len(v) for v in grids.values()]) * ntrials
        pb = ProgressBar(max_value=total_models)
        
        with open(output_file, 'w') as out_fd:
            for prestudy in Grid(self.grids_param['prestudy']):
                prestudy.fit(self.X, self.y, self.z)

                row_mask_tr = np.array([False if i in prestudy.rows else True for i in range(self.xl)])
                col_mask_tr = np.array([False if i in prestudy.features else True for i in range(self.xd)])
                X = self.X[row_mask_tr,:][:,col_mask_tr]
                W = self.X[row_mask_tr,:][:,~col_mask_tr]
                y = self.y[row_mask_tr]
                z = self.z[row_mask_tr]

                for tr_args in itertools.product(*tr_arg_values):
                    tr_kwargs = dict(zip(tr_arg_names, tr_args))
                    tr_bias = tr_kwargs['bias']
                    tr_size = tr_kwargs['size']
                    y_noise = tr_kwargs['y_noise']
                    z_noise = tr_kwargs['z_noise']

                    rand = np.random.RandomState(123456)
                    tr_idx = make_confounding_data(y, z, tr_bias, tr_size, rand)
                    X_tr = X[tr_idx]
                    W_tr = W[tr_idx]
                    y_tr = y[tr_idx]
                    z_tr = z[tr_idx]
                    
                    corr_tr, pval_tr = pearsonr(y_tr, z_tr)
                    
                    if y_noise:
                        y_tr = uniform_noise_1d(y_tr, y_noise, rand)
                    if z_noise:
                        z_tr = uniform_noise_1d(z_tr, z_noise, rand)
                    
                    row_mask_te = np.array([False if i in tr_idx else True for i in range(X.shape[0])])
                    X2 = X[row_mask_te]
                    W2 = W[row_mask_te]
                    y2 = y[row_mask_te]
                    z2 = z[row_mask_te]

                    for te_args in itertools.product(*te_arg_values):
                        te_kwargs = dict(zip(te_arg_names, te_args))
                        te_bias = te_kwargs['bias']
                        te_size = te_kwargs['size']

                        for i_trial in range(ntrials):
                            te_idx = make_confounding_data(y2, z2, te_bias, te_size, rand)
                            X_te = X2[te_idx]
                            W_te = W2[te_idx]
                            y_te = y2[te_idx]
                            z_te = z2[te_idx]
                            corr_te, pval_te = pearsonr(y_te, z_te)
                            
                            for estimator_class, kwargs in Grid(self.grids_param['mainstudy']).get_tuples():
                                if 'prestudy' in inspect.signature(estimator_class).parameters:
                                    kwargs['prestudy'] = prestudy

                                bm_settings = {'estimator': estimator_class.__name__, 'trial': i_trial}
                                bm_settings.update({"estimator_" + k:v for k,v in kwargs.items()})
                                bm_settings.update({"tr_" + k:v for k,v in tr_kwargs.items()})
                                bm_settings.update({"te_" + k:v for k,v in te_kwargs.items()})

                                # save the name of the function or code of the lambda function to the benchmark file
                                if 'cvar_transform' in kwargs and type(kwargs['cvar_transform']) == str:
                                    kwargs['cvar_transform'] = eval(kwargs['cvar_transform'])

                                count += 1
                                pb.update(count)
                                # avoid duplicates if an estimator does not take some parameters (eg BA doesn't have an prestudy parameter)
                                if bm_settings in self.done:
                                    continue

                                self.done.append(bm_settings)

                                clf = estimator_class(**kwargs)

                                fit_map = dict(zip(["X", "W", "y", "z"], [X_tr, W_tr, y_tr, z_tr]))
                                pred_map = dict(zip(["X", "W", "y", "z"], [X_te, W_te, y_te, z_te]))

                                fit_args_names = inspect.signature(clf.fit).parameters.keys()
                                pred_args_names = inspect.signature(clf.predict).parameters.keys()

                                fit_kwargs = {k: fit_map[k] for k in fit_args_names if k in fit_map}
                                pred_kwargs = {k: pred_map[k] for k in pred_args_names if k in pred_map}

                                clf.fit(**fit_kwargs)
                                y_pred = clf.predict(**pred_kwargs)
                                y_prob = clf.predict_proba(**pred_kwargs)
                                
                                output_line = bm_settings.copy()
                                for metric in self.metrics:
                                    score = metric(y_te, y_pred)
                                    output_line[metric.__qualname__] = score
                                
                                prob_true, prob_pred = calibration_curve(y_te, y_prob[:,1], normalize=True)
                                output_line["prob_true"] = prob_true.tolist()
                                output_line["prob_pred"] = prob_pred.tolist()
                                
                                for k in ['estimator_prestudy', 'estimator_cvar_transform']:
                                    if k in output_line:
                                        output_line[k] = str(output_line[k])
                                    
                                output_line['corr_tr'] = corr_tr
                                output_line['pval_tr'] = pval_tr
                                output_line['corr_te'] = corr_te
                                output_line['pval_te'] = pval_te
                                output_line['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                for metric, score in prestudy.scores_raw.items():
                                    output_line["prestudy_%s" % metric] = score
                                
                                out_fd.write("%s\n" % json.dumps(output_line))
                                out_fd.flush()