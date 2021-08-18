import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from time import time
from TimeSeries import TimeSeries
from copy import deepcopy
import os
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from Causal_inference import check_with_original
import importlib
import json
from time import time
import Representation
import threading
import copy

CWD = os.getcwd()
DATASET_NAMES = ["FaceFour", "InlineSkate", "PickupGestureWiimoteZ", "SemgHandMovementCh2"]

TO_IMPORT =  ["mixsd0.1_0.1_causaldb", "mixsd0.1_0.05_causaldb", "mixsd0.2_0.1_causaldb", "mixsd0.2_0.05_causaldb", "randomsd0.1_effectdb", "randomsd0.2_effectdb", "rwalksd0.1_effectdb", "rwalksd0.05_effectdb"]

DATA_PATH = CWD +"/test_files/"

BEST_GAMMA = 5
NEIGHBORS =[2, 5, 10, 100]
PVALS = [0.01, 0.025, 0.05, 0.1]
LAGS = [1, 2]

TAU_MAX = 3 

class DataAttri:
  def __init__(self):
    self.dataset_name = None
    self.import_type = None
    self.alpha_level = None
    self.p_value = None
    self.lagged = None
    self.representation = None
    self.precision = None
    self.recall = None
    self.f_score = None
    self.return_time = None
    self.trueMat = None
    self.pcmci = None
    self.val_matrix = None
    self.link_matrix = None
    self.p_matrix = None
    self.q_matrix = None
    self.compare_matrix = None
    self.var_names = None
    self.model = None


def import_data():
    dataset_dict = {}
    for each in DATASET_NAMES:
        dataset_dict[each] = {}
        dataset_dict[each]["truemat"] = np.load(str(DATA_PATH+each+"_split_truemat.npy"))
        # print(dataset_dict[each]["truemat"])
        dataset_dict[each]["causaldb"] = np.load(str(DATA_PATH+ each+"_causaldb.npy"))
        for import_type in TO_IMPORT:
            dataset_dict[each][import_type] = np.load(str(DATA_PATH+each+"_"+import_type+".npy"))
    return dataset_dict

def process_data_once(dataset_dict):
    importlib.reload(tigramite)
    attr_hold = DataAttri()
    causal = dataset_dict["FaceFour"]["causaldb"]
    # representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = BEST_GAMMA)
    trueMat = dataset_dict["FaceFour"]["truemat"]
    attr_hold.trueMat = trueMat
    effectdb = dataset_dict["FaceFour"]["randomsd0.1_effectdb"]
    effect = effectdb
    var_names = np.arange(len(effectdb))
    df1 = pp.DataFrame(effect.transpose(), datatime = np.arange(len(effect[0])),var_names=var_names)
    df2 = copy.deepcopy(df1)
    parcorr = ParCorr(significance='analytic')
    pcmci1 = PCMCI(
        dataframe=df1, 
        cond_ind_test=parcorr,
        verbosity=0)
    pcmci1.verbosity = 0

    pcmci2 = PCMCI(
        dataframe=df2, 
        cond_ind_test=parcorr,
        verbosity=0)
    pcmci2.verbosity = 0

    attr_hold.dataset_name = "FaceFour"
    attr_hold.import_type = "randomsd0.1_effectdb"
    attr_hold.representation = "non-Grail"
    attr1 = copy.deepcopy(attr_hold)
    attr2 = copy.deepcopy(attr_hold)
    # attr1.pcmci = pcmci1
    # attr2.pcmci = pcmci2
    print("first stage")
    # run_PCMCI(attr1)
    # run_PCMCI_plus(attr2)
    t = time()
    # results = pcmci1.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)
    # q_matrix = attr1.pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
    results = pcmci2.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=None)
    print("second stage done")
    q_matrix = pcmci2.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
    return_time = time() - t
    attr1.return_time = return_time
    attr1.val_matrix = results["val_matrix"]
    attr1.q_matrix = q_matrix
    attr1.p_matrix = results['p_matrix']
    attr1.model = "PCMCI_plus"
    thread_control(attr1)

def process_data(dataset_dict):
    importlib.reload(tigramite)
    counter = 0
    for each in DATASET_NAMES:
        attr_hold = DataAttri()
        causal = dataset_dict[each]["causaldb"]
        representation = Representation.GRAIL(kernel="SINK", d = 100, gamma = BEST_GAMMA)
        trueMat = dataset_dict[each]["truemat"]
        attr_hold.trueMat = trueMat

        for import_type in TO_IMPORT:
            if counter == 1:
                break
            effectdb = dataset_dict[each][import_type]
            n1 = causal.shape[0]
            n2 = effectdb.shape[0]
            # TRAIN_TS, TEST_TS = representation.get_rep_train_test(effectdb, causal)
            # for reps in ["GRAIL", "non-GRAIL"]:     #with and without GRAIL
            #     if reps == "GRAIL":
            #         effect =  TEST_TS
            #     else:
            #         effect = effectdb
            #     # rep = reps
            effect = effectdb
            reps = "non-Grail"
            var_names = np.arange(len(effect))
            attr_hold.var_names = var_names

            attr_hold.dataset_name = each
            attr_hold.import_type = import_type
            attr_hold.representation = reps
            attr1 = copy.deepcopy(attr_hold)
            attr2 = copy.deepcopy(attr_hold)

            df1 = pp.DataFrame(effect.transpose(), datatime = np.arange(len(effect[0])),var_names=var_names)
            df2 = copy.deepcopy(df1)
            parcorr = ParCorr(significance='analytic')
            pcmci1 = PCMCI(
                dataframe=df1, 
                cond_ind_test=parcorr,
                verbosity=0)
            pcmci1.verbosity = 0
            t = time()
            results = pcmci1.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)
            q_matrix = pcmci1.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
            return_time = time() - t
            attr1.return_time = return_time
            attr1.val_matrix = results["val_matrix"]
            attr1.q_matrix = q_matrix
            attr1.p_matrix = results['p_matrix']
            attr1.model = "PCMCI"
            for alpha_level in PVALS:
                link_matrix = pcmci1.return_significant_links(pq_matrix=attr1.q_matrix,
                            val_matrix=attr1.val_matrix, alpha_level=alpha_level)['link_matrix']
                for lagged in LAGS:
                    attrz = copy.deepcopy(attr1)
                    attrz.alpha_level = alpha_level
                    attrz.lagged = lagged
                    attrz.link_matrix = link_matrix
                    z = threading.Thread(target=thread_workers, args=(attrz, ))
                    z.start()

            pcmci2 = PCMCI(
                dataframe=df2, 
                cond_ind_test=parcorr,
                verbosity=0)
            pcmci2.verbosity = 0
            t = time()
            results = pcmci2.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=None)
            q_matrix = pcmci2.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
            return_time = time() - t
            attr2.return_time = return_time
            attr2.val_matrix = results["val_matrix"]
            attr2.q_matrix = q_matrix
            attr2.p_matrix = results['p_matrix']
            attr2.model = "PCMCI_plus"
            for alpha_level in PVALS:
                link_matrix = pcmci1.return_significant_links(pq_matrix=attr2.q_matrix,
                            val_matrix=attr1.val_matrix, alpha_level=alpha_level)['link_matrix']
                for lagged in LAGS:
                    attry = copy.deepcopy(attr2)
                    attry.alpha_level = alpha_level
                    attry.lagged = lagged
                    attry.link_matrix = link_matrix
                    y = threading.Thread(target=thread_workers, args=(attry, ))
                    y.start()
            
            # x = threading.Thread(target=run_PCMCI, args=(attr1, ))
            # x.start()
            # y = threading.Thread(target=run_PCMCI_plus, args=(attr2, ))
            # y.start()
            # counter += 1

def run_PCMCI(attr):
    print("second stage")
    t = time()
    results = attr.pcmci.run_pcmci(tau_max=TAU_MAX, pc_alpha=None)

    q_matrix = attr.pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
    return_time = time() - t
    attr.return_time = return_time
    attr.val_matrix = results["val_matrix"]
    attr.q_matrix = q_matrix
    attr.p_matrix = results['p_matrix']
    attr.model = "PCMCI"
    thread_control(attr)
    pass

def run_PCMCI_plus(attr):
    print("plus second stage")
    t = time()
    print(attr.pcmci.dataframe.values.shape)
    results = attr.pcmci.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=None)
    # try:
    #     results = attr.pcmci.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=None)
    # except:
    #     print(attr.pcmci.dataframe.values.shape)
    #     with open(f'{CWD}/model_results/2.npy', "wb") as f:
    #         np.save(f, attr.pcmci.dataframe)
    #     pass

    q_matrix = attr.pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=8, fdr_method='fdr_bh')
    return_time = time() - t
    attr.return_time = return_time
    attr.val_matrix = results["val_matrix"]
    attr.q_matrix = q_matrix
    attr.p_matrix = results['p_matrix']
    attr.model = "PCMCI_plus"
    thread_control(attr)
    pass

def thread_control(attr):
    for alpha_level in PVALS:
        link_matrix = attr.pcmci.return_significant_links(pq_matrix=attr.q_matrix,
                    val_matrix=attr.val_matrix, alpha_level=alpha_level)['link_matrix']
        attr.link_matrix = link_matrix
        for lagged in LAGS:
            attrz = copy.deepcopy(attr)
            attrz.alpha_level = alpha_level
            attrz.lagged = lagged
            attrz.link_matrix = link_matrix
            z = threading.Thread(target=thread_workers, args=(attrz, ))
            z.start()


def thread_workers(attr):
    compare_matrix = []
    val_matrix = []
    for i in range(len(attr.link_matrix)):
        place_holder_link = []
        place_holder_val = []
        for j in range(len(attr.link_matrix[i])):
            element1 = [attr.link_matrix[i][j][attr.lagged]]
            element2 = [attr.val_matrix[i][j][attr.lagged]]
            place_holder_link.append(element1)
            place_holder_val.append(element2)
        compare_matrix.append(place_holder_link)
        val_matrix.append(place_holder_val)
    compare_matrix = np.array(compare_matrix)
    val_matrix = np.array(val_matrix)
    check_matrix = compare_matrix.reshape((attr.trueMat.shape[0], attr.trueMat.shape[1]))
    check_results = check_with_original(attr.trueMat, check_matrix)
    attr.precision = check_results[0]
    attr.recall = check_results[1]
    attr.f_score = check_results[2]
    attr.compare_matrix = compare_matrix
    # q = threading.Thread(target=time_series_plot, args=(attr, ))
    # q.start()
    # e = threading.Thread(target=process_graph_plot, args=(attr, ))
    # e.start()
    t = threading.Thread(target=saving_matrices, args=(attr, ))
    t.start()
    s = threading.Thread(target=saving_attributes, args=(attr, ))
    s.start()

def time_series_plot(attr):
    tp.plot_time_series_graph(
        figsize=(200, 200),
        node_size=0.003,
        val_matrix=attr.val_matrix,
        link_matrix=attr.compare_matrix,
        var_names=attr.var_names,
        link_colorbar_label='MCI',
        )
    plt.savefig(f"{CWD}/model_results/{attr.dataset_name}_{attr.import_type}_\
        P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_time_series_graph.png")
    plt.clf()
    plt.close()
    pass

def process_graph_plot(attr):
    tp.plot_graph(
        val_matrix=attr.val_matrix,
        link_matrix=attr.compare_matrix,
        var_names=attr.var_names,
        figsize=(40, 40),
        link_colorbar_label='cross-MCI (edges)',
        node_colorbar_label='auto-MCI (nodes)',
        )
    plt.savefig(f"{CWD}/model_results/{attr.dataset_name}_{attr.import_type}_\
        P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_process_graph.png")
    plt.clf()
    plt.close()
    pass

def saving_matrices(attr):
    with open(f'{CWD}/model_results/{attr.dataset_name}_{attr.import_type}_\
        P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_val_matrices_PCMCI.npy', 'wb') as f:
        np.save(f, attr.val_matrix)

    with open(f'{CWD}/model_results/{attr.dataset_name}_{attr.import_type}_\
        P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_link_matrices.npy', 'wb') as f:
        np.save(f, attr.compare_matrix)

    with open(f'{CWD}/model_results/{attr.dataset_name}_{attr.import_type}_\
        P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_pval_matrices.npy', 'wb') as f:
        np.save(f, attr.p_matrix)

def saving_attributes(attr):
    results_dict = {}
    results_dict["data_examined"] = attr.dataset_name
    results_dict["effect_used"] = attr.import_type
    results_dict["alpha-level"] = attr.alpha_level
    results_dict["lag value"] = attr.lagged
    results_dict["representation used"] = attr.representation
    results_dict["precision"]= attr.precision
    results_dict["recall"]= attr.recall
    results_dict["f-score"]= attr.f_score
    results_dict["return_time"] = attr.return_time
    with open(f'{CWD}/model_results/{attr.dataset_name}_{attr.import_type}_\
        P{attr.alpha_level}_L{attr.lagged}_{attr.representation}_{attr.model}_compared_stats.json', 'w') as f:
        json.dump(results_dict, f)

def main():
    dataset_dict = import_data()
    process_data_once(dataset_dict)
    # process_data(dataset_dict)
    pass

if __name__ == "__main__":
    main()