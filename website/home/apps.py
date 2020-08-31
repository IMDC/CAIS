from django.apps import AppConfig

class ActiveLearnerConfig(AppConfig):
    name = "home"
    verbose_name = "ActiveLearner"

    def ready(self):
        self.count = 0
        from .alc import ActiveLearningClient
        self.act_model = ActiveLearningClient() # parameter in.

    def set_x_pool(self):
        print("Set X pool for hearing group")
        self.act_model.get_data_for_hearing_group()
        
    def make_prediction(self):
        self.query_idx, self.preds, self.queried_val = self.act_model.make_preds()
        return (self.query_idx, self.preds, self.queried_val)

    def learn_ratings(self, client_list):
        self.learned_model = self.act_model.train_learner(self.query_idx, client_list)
        self.count += 1
        print("LEARN RATINGS COUNT:", self.count)        
        return self.learned_model

    def get_prediction(self):
        return (self.query_idx, self.preds, self.queried_val)
