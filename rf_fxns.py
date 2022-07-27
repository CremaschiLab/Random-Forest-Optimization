import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib
from pyomo.environ import *
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import time


def calc_rmse(z, zhat):
    z = np.asarray(z)
    zhat = np.asarray(zhat)
    N = len(z)

    sse = sum((z - zhat) ** 2)
    rmse = (sse / N) ** 0.5
    return (rmse)


def r_squared(z, zhat, s_size, m_size):
    z = np.asarray(z)
    zhat = np.asarray(zhat)

    zavg = np.average(z)
    sstot = sum((z - zavg) ** 2)
    sse = sum((z - zhat) ** 2)
    r2 = 1 - (sse / sstot)
    adjr2 = 1 - ((1 - r2) * (s_size - 1) / (s_size - m_size - 1))
    return (r2, adjr2)


def rf_thresholds(forest):
    leaves = 0
    tholds = 0

    for estimator in forest.estimators_:
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        for i in range(n_nodes):
            if is_leaves[i]:
                leaves += 1

            else:
                tholds += 1
    avg_leaves = leaves / len(forest.estimators_)
    avg_tholds = tholds / len(forest.estimators_)
    return (avg_tholds)

def train_RF(xtrain,ytrain,filename='rf_mode',random_state=25):
    ##xtrain: training inputs (numpy array)
    ##ytrain: training outputs (numpy array(
    ##filename: location for training logfile
    ##random_state: random seed for rf training, keep this value the same if you want to get the same model with each training

    dim=len(xtrain[0])
    file = open(filename+'.txt', 'w')

    ymax = max(ytrain)

    delta = 1000
    ntrees = int(50 + (dim - 15) - 1)
    max_trees = 150
    last5 = []
    last_avg = 1000
    terror = {}
    last_delta = 1000

    ntrials = 20
    tsize = 0.20

    train = True
    start_time = time.time()
    while train:
        errors = []
        ntrees += 1
        f = 0
        while f < ntrials:
            model = RandomForestRegressor(n_estimators=ntrees, random_state=random_state)

            x_train, x_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size=tsize)
            model.fit(x_train, y_train)
            yhat = model.predict(x_val)
            ei = (calc_rmse(y_val, yhat)) / (ymax - ymin)
            errors.append(ei)

            f += 1

        e_avg = np.average(errors)
        terror[ntrees] = e_avg

        if len(last5) >= 5:
            last5.remove(last5[0])

        last5.append(e_avg)
        last5_avg = np.average(last5)

        delta = np.abs((last5_avg - last_avg) / last_avg)

        if ntrees == max_trees:
            delta = 0
            last_delta = 0

        last_avg = last5_avg
        # print('Trees: '+str(ntrees)+', nRMSE: '+str(e_avg)+', Average of Last 5 Errors: '+str(last5_avg)+', Delta: '+str(delta))
        t_str = 'Trees: ' + str(ntrees) + ', nRMSE: ' + str(e_avg) + ', Average of Last 5 Errors: ' + str(
            last5_avg) + ', Delta: ' + str(delta)
        file.write(t_str + '\n')

        if (delta < level and last_delta < level):
            train = False
        last_delta = delta
        if ntrees < 30:
            train = True
        if e_avg < 0.001:
            train = False

    ntrees_f = min(terror, key=terror.get)
    model_f = RandomForestRegressor(n_estimators=ntrees_f, random_state=44)
    model_f.fit(xtrain, ytrain)
    end_time = time.time()

    train_time = end_time - start_time

    model_size = rf_thresholds(model_f)


    file.write('*' * len(t_str) + '\n' + '\n')
    file.write('Training Time: ' + str(train_time) + ' sec' + '\n')
    file.write('Model Size: ' + str(model_size) + '\n')
    file.write('Trees: ' + str(len(model_f.estimators_)) + '\n')
    file.close()

    ##model_f: trained random forest model
    ##train_time: time (seconds) to train model
    return(model_f,train_time)


def solve_RF(forest, dim, ub, lb):

    ##forest: trained random forest model (sklearn randomforestregressor)
    ##dim: number of input dimensions
    ##ub: upper bounds on decision variables as dictionary, ex - {1: 2.5, 2: 10.1}
    ##lb: lower bounds on decision variables as dictionary, ex - {1: 1.5, 2: 2.1}



    no_trees = len(forest.estimators_)

    leaves = []
    parents = []
    tholds = {}

    nodes = []
    leaf_val = {}
    node_tup = []
    lr_n = []

    first_node = []
    l_order = []
    t = 0

    while t < no_trees:
        estimator = forest.estimators_[t]
        tind = t + 1

        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))

            else:
                is_leaves[node_id] = True


        f_n = []
        f_n.append(tind)
        f_n.append(1)
        first_node.append(tuple(f_n))

        for i in range(n_nodes):
            if is_leaves[i]:
                ##tind_lind ---> (tree index, leaf index)
                tind_lind = []
                tind_lind.append(tind)
                tind_lind.append(i + 1)
                leaves.append(tuple(tind_lind))
                nodes.append(tuple(tind_lind))

                leaf_val_i = estimator.tree_.value[i][0][0]

                leaf_val[tuple(tind_lind)] = leaf_val_i
                l_order.append(i + 1)

            else:

                tind_pind = []
                tind_pind.append(tind)
                tind_pind.append(i + 1)
                parents.append(tuple(tind_pind))
                nodes.append(tuple(tind_pind))

                tholds[tuple(tind_pind)] = threshold[i]

                tind_pind_rind_xind = []
                tind_pind_rind_xind.append(tind)
                tind_pind_rind_xind.append(i + 1)
                tind_pind_rind_xind.append(children_right[i] + 1)
                tind_pind_rind_xind.append(feature[i] + 1)
                node_tup.append(tuple(tind_pind_rind_xind))

                tind_pind_lind_rind = []
                tind_pind_lind_rind.append(tind)
                tind_pind_lind_rind.append(i + 1)
                tind_pind_lind_rind.append(children_left[i] + 1)
                tind_pind_lind_rind.append(children_right[i] + 1)
                lr_n.append(tuple(tind_pind_lind_rind))


        t = t + 1

    avg_hold = len(tholds) / len(forest.estimators_)
    start_time = time.time()
    model = ConcreteModel()

    model.N = Set(initialize=nodes)
    model.F = RangeSet(1, dim)
    model.P = Set(initialize=parents)
    model.L = Set(initialize=leaves)
    model.val = Param(model.L, initialize=leaf_val)
    model.thrs = Param(model.P, initialize=tholds)

    model.lrtup = Set(initialize=lr_n)
    model.ntup = Set(initialize=node_tup)
    model.node1 = Set(initialize=first_node)

    model.y = Var(model.N, domain=Binary)

    def fb(model, i):
        return (lb[i], ub[i])

    model.x = Var(model.F, domain=Reals, bounds=fb)

    def xmax(i):
        return (ub[i])

    def xmin(i):
        return (lb[i])

    def obj_expression(model):
        return ((sum(model.val[t, l] * model.y[t, l] for (t, l) in model.L)) * (1 / no_trees))

    model.OBJ = Objective(rule=obj_expression, sense=minimize)



    def node1_rule(model, t, l):
        return 1 == model.y[t, l]

    model.node1Constraint = Constraint(model.node1, rule=node1_rule)

    def plr_rule(model, h, i, j, k):
        return model.y[h, i] == model.y[h, j] + model.y[h, k]

    model.plrConstraint = Constraint(model.lrtup, rule=plr_rule)

    def branch_rule1(model, h, i, j, k):
        return model.x[k] >= model.thrs[h, i] - ((xmax(k) - xmin(k)) * (1 - model.y[h, j])) - (xmax(k) - xmin(k)) * (
                1 - model.y[h, i]) + 0.0002 * model.y[h, j]

    model.branchConstraint1 = Constraint(model.ntup, rule=branch_rule1)

    def branch_rule2(model, h, i, j, k):
        return model.x[k] <= model.thrs[h, i] + ((xmax(k) - xmin(k)) * model.y[h, j]) + (xmax(k) - xmin(k)) * (
                1 - model.y[h, i])

    model.branchConstraint2 = Constraint(model.ntup, rule=branch_rule2)

    opt = SolverFactory('cplex',executable='<insert cplex solver file location>')

    results = opt.solve(model, load_solutions=True)

    end_time = time.time()
    model.solutions.store_to(results)

    solve_time = end_time - start_time
    is_inf = str(results).find('infeasible')
    if is_inf != -1:
        solve_time = 0
        z = np.zeros(dim)
        obj = 'infeasible'
        results = 'none'
    else:

        z = []
        i = 1
        while i <= dim:
            z.append(model.x[i].value)
            # print(model.x[i].value)
            i = i + 1

        z = np.asarray(z)

        obj = results['Solution'][0]['Objective']['OBJ']['Value']
    ##z: optimum location
    ##obj: optimum objective function value
    ##solve_time: solution time (seconds)
    ##results: pyomo/cplex results file
    return (z, obj,solve_time,results)
