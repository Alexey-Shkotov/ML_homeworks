import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=1):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)
    :param min_samples_leaf: int, ограничение снизу на кол-во элементов в левом и правом поддереве

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if type(feature_vector) != np.array:
        feature_vector = np.array(feature_vector)
    if type(target_vector) != np.array:
        target_vector = np.array(target_vector)
    feature_vals_sorted_idx = np.argsort(feature_vector)
    feature_vals_sorted = feature_vector[feature_vals_sorted_idx]
    unique_feature_vals, unique_feature_vals_idx = np.unique(feature_vals_sorted, return_index=True)
    target_sorted = target_vector[feature_vals_sorted_idx]

    # compute thresholds
    thresholds = (unique_feature_vals[:-1] + unique_feature_vals[1:]) / 2.

    # compute gini values
    target_cumsums = np.cumsum(target_sorted)
    # calculate number of elements
    elements_number = len(feature_vector)
    # calculate number of elements in each subtree for each split
    elements_number_left_subtree = unique_feature_vals_idx[1:]  # as unique_feature_vals_idx[0] == 0, so if we take it, we will have a split with empty left subtree (and they are banned)
    elements_number_right_subtree = elements_number - elements_number_left_subtree
    # calculate number of positive class elements in each subtree for each split
    pos_elements_number = target_cumsums[-1]
    pos_elements_number_left_subtree = target_cumsums[elements_number_left_subtree - 1]
    pos_elements_number_right_subtree = pos_elements_number - pos_elements_number_left_subtree
    # calculate probabilities for each split
    p1_left_subtree = pos_elements_number_left_subtree / elements_number_left_subtree
    p0_left_subtree = 1. - p1_left_subtree
    p1_right_subtree = pos_elements_number_right_subtree / elements_number_right_subtree
    p0_right_subtree = 1. - p1_right_subtree
    # calculate H(R_l) and H(R_r) for each split
    H_left_gini = 1. - np.square(p0_left_subtree) - np.square(p1_left_subtree)
    H_right_gini = 1. - np.square(p0_right_subtree) - np.square(p1_right_subtree)
    # calculate Q(R) for each split
    ginis = (-(elements_number_left_subtree / elements_number) * H_left_gini -
             (elements_number_right_subtree / elements_number) * H_right_gini)

    if min_samples_leaf == 1:
        # find the best threshold (by gini values) and its gini value
        first_best_gini_idx = np.argmax(ginis)
        threshold_best = thresholds[first_best_gini_idx]
        gini_best = ginis[first_best_gini_idx]
    else:
        # look for such thresholds and gini values that satisfy the min_samples_leaf condition
        satisfying_ginis = ginis[(elements_number_left_subtree >= min_samples_leaf) & (elements_number_right_subtree >= min_samples_leaf)]
        satisfying_thresholds = thresholds[(elements_number_left_subtree >= min_samples_leaf) & (elements_number_right_subtree >= min_samples_leaf)]
        if len(satisfying_thresholds) == 0:
            threshold_best = None
            gini_best = None
        else:
            # find the best threshold (by gini values) and its gini value
            first_best_gini_idx = np.argmax(satisfying_ginis)
            threshold_best = satisfying_thresholds[first_best_gini_idx]
            gini_best = satisfying_ginis[first_best_gini_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._depth = 0

    def _fit_node(self, sub_X, sub_y, node, node_depth=0):
        node["depth"] = node_depth
        if np.all(sub_y == sub_y[0]) or (self._max_depth is not None and node["depth"] == self._max_depth) or (self._min_samples_split is not None and len(sub_X) < self._min_samples_split):
            # ^--- here we check that each element is of the same class => we put "==" instead of "!="
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            self._depth = max(self._depth, node["depth"])
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):  # start from 0, not 1
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # here I made "current_click / current_count"
                    # instead of "current_count / current_click"
                    # as we should find the ratio of positive objects in the category to all objects in the category
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                # ^--"lambda x: x[0]" instead of "lambda x: x[1]" to get the categories, not the ratios
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))  # list(map(...)) I was looking for it for 3 hours...
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue
            # changed the if-clause above as it didn't make any sense;
            # now it means that if there is only one unique element in feature_vector in this iteration,
            # so we can't find any threshold,
            # then we skip this iteration

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = (feature_vector < threshold)

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":  # changed "Categorical" to "categorical"
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            # changed "(...).most_common(1)" to "(...).most_common(1)[0][0]" to make this expression return a class,
            # not something else
            self._depth = max(self._depth, node["depth"])
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], node["depth"] + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], node["depth"] + 1)
        # we should get "np.logical_not(split)" from sub_y, not "split"

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            threshold_best = node["threshold"]
            if x[feature_best] < threshold_best:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_best] == "categorical":
            categories_left = node["categories_split"]
            if x[feature_best] in categories_left:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_depth(self):
        return self._depth

    def get_params(self, deep=False):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }