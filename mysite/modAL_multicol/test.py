









from modAL_multimodal.models import Committee
from modAL_multimodal.disagreement import vote_entropy_sampling








# learner1 = ActiveLearner(
#     estimator=RandomForestClassifier(),
#     query_strategy=random_sampling,
#     X_training=X_training, y_training=y_training
# )

# learner2 = ActiveLearner(
#     estimator=RandomForestClassifier(),
#     query_strategy=random_sampling,
#     X_training=X_training, y_training=y_training
# )

# # a list of ActiveLearners:
# learners = [learner_1, learner_2]

# committee = Committee(
#     learner_list=learners,
#     query_strategy=vote_entropy_sampling
# )