from typing import Any, Callable, List, Optional, Tuple
from tensorflow.keras.models import Sequential
import numpy as np
from modAL_multicol.acquisition import max_EI
from modAL_multicol.disagreement import max_std_sampling, vote_entropy_sampling
from modAL_multicol.models.base import BaseCommittee, BaseLearner
from modAL_multicol.uncertainty import uncertainty_sampling
from modAL_multicol.utils.data import data_vstack, modALinput, retrieve_rows
from modAL_multicol.utils.validation import check_class_labels, check_class_proba
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y

"""
Classes for active learning algorithms
--------------------------------------
"""


class ActiveLearner(BaseLearner):
    """
    This class is an model of a general classic (machine learning) active learning algorithm.

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
            which the model has been trained on. If provided, the method fit() of estimator is called during __init__()
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
        super().__init__(estimator, query_strategy, **fit_kwargs)

        self.X_training = X_training
        self.y_training = y_training

        if X_training is not None:
            self._fit_to_known(bootstrap=bootstrap_init, **fit_kwargs)

    def _add_training_data(self, X: modALinput, y: modALinput) -> None:
        """
        Adds the new data and label to the known data, but does not retrain the model.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.

        Note:
            If the classifier has been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        check_X_y(
            X,
            y,
            accept_sparse=True,
            ensure_2d=False,
            allow_nd=True,
            multi_output=True,
            dtype=None,
            force_all_finite=self.force_all_finite,
        )

        if self.X_training is None:
            self.X_training = X
            self.y_training = y
        else:
            try:
                self.X_training = data_vstack((self.X_training, X))
                self.y_training = data_vstack((self.y_training, y))
            except ValueError:
                raise ValueError(
                    "the dimensions of the new training data and label must"
                    "agree with the training data and labels provided so far"
                )

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> "BaseLearner":
        """
        Fits self.estimator to the training data and labels provided to it so far.

        Args:
            bootstrap: If True, the method trains the model on a set bootstrapped from the known training instances.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """        

        if not bootstrap:
            self.estimator.fit(self.X_training, self.y_training, **fit_kwargs)            
        else:
            n_instances = self.X_training.shape[0]
            bootstrap_idx = np.random.choice(
                range(n_instances), n_instances, replace=True
            )
            self.estimator.fit(
                self.X_training[bootstrap_idx],
                self.y_training[bootstrap_idx],
                **fit_kwargs
            )

        return self

    def fit(
        self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs
    ) -> "BaseLearner":
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied data, then stores it
        internally for the active learning loop.

        Args:
            X: The samples to be fitted.
            y: The corresponding labels.
            bootstrap: If true, trains the estimator on a set bootstrapped from X.
                Useful for building Committee models with bagging.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Note:
            When using scikit-learn estimators, calling this method will make the ActiveLearner forget all training data
            it has seen!

        Returns:
            self
        """
        check_X_y(
            X,
            y,
            accept_sparse=True,
            ensure_2d=False,
            allow_nd=True,
            multi_output=True,
            dtype=None,
            force_all_finite=self.force_all_finite,
        )
        self.X_training, self.y_training = X, y
        return self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)

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
        if not only_new:
            self._add_training_data(X, y)
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            check_X_y(
                X,
                y,
                accept_sparse=True,
                ensure_2d=False,
                allow_nd=True,
                multi_output=True,
                dtype=None,
                force_all_finite=self.force_all_finite,
            )
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)


"""
Classes for committee based algorithms
--------------------------------------
"""


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

    def save_model(self, *filenames):
        """
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
        Checks the known class labels by each learner, merges the labels and returns a mapping which maps the learner's
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

    def fit(self, X: modALinput, y: modALinput, **fit_kwargs) -> "BaseCommittee":
        """
        Fits every learner to a subset sampled with replacement from X. Calling this method makes the learner forget the
        data it has seen up until this point and replaces it with X! If you would like to perform bootstrapping on each
        learner using the data it has seen, use the method .rebag()!
        Calling this method makes the learner forget the data it has seen up until this point and replaces it with X!
        Args:
            X: The samples to be fitted on.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().fit(X, y, **fit_kwargs)
        self._set_classes()

    def teach(
        self,
        X: modALinput,
        y: modALinput,
        bootstrap: bool = False,
        only_new: bool = False,
        **fit_kwargs
    ) -> None:
        """
        Adds X and y to the known training data for each learner and retrains learners with the augmented dataset.
        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        super().teach(X, y, bootstrap=bootstrap, only_new=only_new, **fit_kwargs)
        self._set_classes()

    def predict(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the class of the samples by picking the consensus prediction.
        Args:
            X: The samples to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the :meth:`predict_proba` of the Committee.
        Returns:
            The predicted class labels for X.
        """
        # getting average certainties
        proba = self.predict_proba(X, **predict_proba_kwargs)

        if proba.shape[1] > 4:  # multi output flatten, proba.shape[1] == 20
            preds = np.split(proba[0], int(proba.shape[1] / self.n_classes_))
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
            print("preds after argmax@predict in learners.py", fin_preds)
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
        return np.mean(self.vote_proba(X, **predict_proba_kwargs), axis=1)

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
        Predicts the labels for the supplied data for each learner in the Committee.
        Args:
            X: The samples to cast votes.
            **predict_kwargs: Keyword arguments to be passed to the :meth:`predict` of the learners.
        Returns:
            The predicted class for each learner in the Committee and each sample in X.
        """
        n_learners = len(self.learner_list)
        prediction = np.zeros(
            shape=(X.shape[0], n_learners * 20)
        )  # in the shape of multiple columns, put next to each other

        p_idx = 0  # initialization for padding
        for learner_idx, learner in enumerate(self.learner_list):
            tmp_prediction = learner.predict(X, **predict_kwargs)

            if isinstance(
                learner.estimator, Sequential
            ):  # check if the learner.estimator is a Keras model
                splited_y = np.hsplit(
                    tmp_prediction, 4
                )  # because we only have four quality factors atm...
                rating_vals = np.array(
                    list(map(lambda x: np.argmax(x, axis=1), splited_y))
                )
                y_classes = (
                    rating_vals.transpose()
                )  # now let's stack up the splited list, indices of rating vote
                prediction = (
                    y_classes
                    if learner_idx == 0
                    else np.hstack((prediction, y_classes))
                )
        return prediction

    def vote_proba(self, X: modALinput, **predict_proba_kwargs) -> Any:
        """
        Predicts the probabilities of the classes for each sample and each learner.
        Args:
            X: The samples for which class probabilities are to be calculated.
            **predict_proba_kwargs: Keyword arguments for the :meth:`predict_proba` of the learners.
        Returns:
            Probabilities of each class for each learner and each instance.
        """       
        proba = np.zeros(shape=(X.shape[0], len(self.learner_list), 20))
        for learner_idx, learner in enumerate(self.learner_list):
            tmp_p = learner.predict_proba(X, **predict_proba_kwargs)
            tmp_t = tmp_p.transpose()
            proba[:, learner_idx, :] = tmp_p
        return proba
