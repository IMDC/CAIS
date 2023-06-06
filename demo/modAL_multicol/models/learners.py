import numpy as np

np.set_printoptions(precision=3)

import os
import datetime

from tensorflow.keras.models import Sequential

from typing import Callable, Optional, Tuple, List, Any

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from modAL_multicol.models.base import BaseLearner, BaseCommittee
from modAL_multicol.utils.validation import check_class_labels, check_class_proba
from modAL_multicol.utils.data import modALinput
from modAL_multicol.uncertainty import uncertainty_sampling
from modAL_multicol.disagreement import vote_entropy_sampling


"""
Classes for active learning algorithms
--------------------------------------
"""


class ActiveLearner(BaseLearner):
    """
    This class is an abstract model of a general active learning algorithm.

    Args:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, modAL.uncertainty.uncertainty_sampling.
        X_training: Initial training samples, if available.
        y_training: Initial training labels corresponding to initial training samples.
        bootstrap_init: If initial training data is available, bootstrapping can be done during the first training.
            Useful when building Committee models with bagging.
        **fit_kwargs: keyword arguments.

    Attributes:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop.
        X_training: If the model hasn't been fitted yet it is None, otherwise it contains the samples
            which the model has been trained on.
        y_training: The labels corresponding to X_training.

    Examples:

        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from modAL.models import ActiveLearner
        >>> iris = load_iris()
        >>> # give initial training examples
        >>> X_training = iris['data'][[0, 50, 100]]
        >>> y_training = iris['target'][[0, 50, 100]]
        >>>
        >>> # initialize active learner
        >>> learner = ActiveLearner(
        ...     estimator=RandomForestClassifier(),
        ...     X_training=X_training, y_training=y_training
        ... )
        >>>
        >>> # querying for labels
        >>> query_idx, query_sample = learner.query(iris['data'])
        >>>
        >>> # ...obtaining new labels from the Oracle...
        >>>
        >>> # teaching newly labelled examples
        >>> learner.teach(
        ...     X=iris['data'][query_idx].reshape(1, -1),
        ...     y=iris['target'][query_idx].reshape(1, )
        ... )
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        query_strategy: Callable = uncertainty_sampling,
        X_training: Optional[modALinput] = None,
        y_training: Optional[modALinput] = None,
        bootstrap_init: bool = False,
        **fit_kwargs
    ) -> None:
        super().__init__(
            estimator,
            query_strategy,
            X_training,
            y_training,
            bootstrap_init,
            **fit_kwargs
        )

    def teach(
        self,
        X: modALinput,
        y: modALinput,
        bootstrap: bool = False,
        only_new: bool = False,
        **fit_kwargs
    ) -> None:
        """
        Adds X and y to the known training data and retrains the predictor with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, training is done on a bootstrapped dataset. Useful for building Committee models
                with bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
                Useful when working with models where the .fit() method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._add_training_data(X, y)

        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)


class Committee(BaseCommittee):
    """
    This class is an abstract model of a committee-based active learning algorithm.

    Args:
        learner_list: A list of ActiveLearners forming the Committee.
        query_strategy: Query strategy function. Committee supports disagreement-based query strategies from
            :mod:`modAL.disagreement`, but uncertainty-based ones from :mod:`modAL.uncertainty` are also supported.

    Attributes:
        classes_: Class labels known by the Committee.
        n_classes_: Number of classes known by the Committee.

    Examples:

        >>> from sklearn.datasets import load_iris
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from modAL.models import ActiveLearner, Committee
        >>>
        >>> iris = load_iris()
        >>>
        >>> # initialize ActiveLearners
        >>> learner_1 = ActiveLearner(
        ...     estimator=RandomForestClassifier(),
        ...     X_training=iris['data'][[0, 50, 100]], y_training=iris['target'][[0, 50, 100]]
        ... )
        >>> learner_2 = ActiveLearner(
        ...     estimator=KNeighborsClassifier(n_neighbors=3),
        ...     X_training=iris['data'][[1, 51, 101]], y_training=iris['target'][[1, 51, 101]]
        ... )
        >>>
        >>> # initialize the Committee
        >>> committee = Committee(
        ...     learner_list=[learner_1, learner_2]
        ... )
        >>>
        >>> # querying for labels
        >>> query_idx, query_sample = committee.query(iris['data'])
        >>>
        >>> # ...obtaining new labels from the Oracle...
        >>>
        >>> # teaching newly labelled examples
        >>> committee.teach(
        ...     X=iris['data'][query_idx].reshape(1, -1),
        ...     y=iris['target'][query_idx].reshape(1, )
        ... )
    """

    def __init__(
        self,
        learner_list: List[ActiveLearner],
        given_classes=None,
        query_strategy: Callable = vote_entropy_sampling,
    ) -> None:
        super().__init__(learner_list, query_strategy)
        self._set_classes(given_classes)
        self.queried_X = {}

    def save_model(self, *filenames):
        """
        #edited
        export estimators in the committee to files h5 in the given directory
        """
        try:
            # print(filenames) # range of argument values ~ keys/filenames
            for l_idx in range(len(self.learner_list)):
                self.learner_list[l_idx].estimator.save(filenames[l_idx])
        except AttributeError as e:
            print("We got an error, the estimator did not save: ", e)

    def _set_classes(self, given_classes=None):
        """
        #edited
        Checks the known class labels by each learner,
        merges the labels and returns a mapping which maps the learner's
        classes to the complete label list.
        """
        if isinstance(self.learner_list[0].estimator, Sequential):
            self.classes_ = np.array([0, 1, 2, 3, 4])

        else:
            if given_classes is None:  # class definition not given
                # assemble the list of known classes from each learner
                try:
                    # if estimators are fitted
                    known_classes = tuple(
                        learner.estimator.classes_ for learner in self.learner_list
                    )
                    conca = np.concatenate(known_classes)
                    while conca[0].ndim > 0:  # handle when given has more dimension
                        conca = np.concatenate(conca)
                except AttributeError:
                    # handle unfitted estimators
                    self.classes_ = None
                    self.n_classes_ = 0
                    return
                except ValueError:
                    conca = [c for t in known_classes for c in t]
                    conca = np.concatenate(conca)
                finally:
                    self.classes_ = np.unique(conca, axis=0)
            else:
                self.classes_ = given_classes
        self.n_classes_ = len(self.classes_)

    def _add_training_data(self, X: modALinput, y: modALinput):
        super()._add_training_data(X, y)
        if not isinstance(self.learner_list[0].estimator, Sequential):
            self._set_classes()

    def predict(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        #edited
        Predicts the class of the samples by picking the consensus prediction.
        in range (0-4)

        Args:
            X: The samples to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            The predicted class labels for X.
        """
        # getting average certainties
        proba = self.predict_proba(X, **predict_proba_kwargs)
        # print(proba)

        if proba.shape[1] > 4:  # multi output flatten, proba.shape[1] == 20
            preds = np.split(proba[0], int(proba.shape[1] / self.n_classes_))

            # print("preds@predict in learners.py", preds)

            fin_preds = []
            for c in range(len(preds)):
                col = preds[c]
                rate_idx = np.argwhere(col > 0.5)
                # 0.5 defined by the paper Cheng, J., Wang, Z., & Pollastri, G. (2008, June).
                # A neural network approach to ordinal regression.
                # In 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence) (pp. 1279-1284). IEEE.
                if rate_idx.size == 0:
                    fin_preds.append(
                        np.argmax(col)
                    )  # regular implementation for data format with one-hot coding and so
                else:
                    fin_preds.append(rate_idx[-1][0])  # the paper implmenetation
            # finding the sample-wise max probability
            # print("preds after argmax@predict in learners.py", fin_preds)
            return fin_preds
        else:
            max_proba_idx = np.argmax(
                proba, axis=1
            )  # finding the sample-wise max probability
            return self.classes_[max_proba_idx]  # translating label indices to labels

    def predict_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Consensus probabilities of the Committee.

        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.

        Returns:
            Class probabilities for X.
        """
        v_proba = self.vote_proba(X, **predict_proba_kwargs)
        return np.mean(v_proba, axis=1)

    def score(
        self, X: modALinput, y: modALinput, sample_weight: List[float] = None
    ) -> Any:
        """
        Returns the mean accuracy on the given test data and labels.

        Todo:
            Why accuracy?

        Args:
            X: The samples to score.
            y: Ground truth labels corresponding to X.
            sample_weight: Sample weights.

        Returns:
            Mean accuracy of the classifiers.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def vote(self, X: modALinput, **predict_kwargs) -> Any:
        """
        #edited for tensorflow version 2.0
        Predicts the labels for the supplied data for each learner in the Committee.

        Args:
            X: The samples to cast votes.
            **predict_kwargs: Keyword arguments to be passed to the :meth:`predict` of the learners.

        Returns:
            The predicted class for each learner in the Committee and each sample in X.
        """
        np.set_printoptions(suppress=True)

        n_learners = len(self.learner_list)
        prediction = np.zeros(
            shape=(X.shape[0], n_learners * 20)
        )  # in the shape of multiple columns, put next to each other
        p_idx = 0  # initialization for padding
        for learner_idx, learner in enumerate(self.learner_list):
            tmp_prediction = learner.predict(X, **predict_kwargs)
            splited_y = np.hsplit(
                tmp_prediction, 4
            )  # because we only have four quality factors atm...
            rating_vals = np.array(
                list(
                    map(lambda x: np.apply_along_axis(self.get_rating, 1, x), splited_y)
                )
            )
            y_classes = (
                rating_vals.transpose()
            )  # now let's stack up the splited list, indices of rating vote
            prediction = (
                y_classes if learner_idx == 0 else np.hstack((prediction, y_classes))
            )
        # print(prediction)
        return prediction

    def get_rating(self, x):
        """
        given [1, 0.8, 0.6, 0, 0] returns 2
        """
        idx = 0
        for i in range(len(x)):
            # if x[i] == 1:
            if x[i] > 0.5:
                idx = i
        return idx

    def vote_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        #edited
        Predicts the probabilities of the classes for each sample and each learner.

        Args:
            X: The samples for which class probabilities are to be calculated.
            **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the learners.

        Returns:
            Probabilities of each class for each learner and each instance.
        """
        proba = np.zeros(shape=(X.shape[0], len(self.learner_list), 20))
        for learner_idx, learner in enumerate(self.learner_list):
            tmp_p = np.round(learner.predict_proba(X, **predict_proba_kwargs))
            tmp_t = tmp_p.transpose()
            proba[:, learner_idx, :] = tmp_p
        # np.set_printoptions(suppress=True)
        # print(proba)
        return proba
