from django.apps import AppConfig

class ActiveLearnerConfig(AppConfig):
    name = "home"
    verbose_name = "ActiveLearner"

    def ready(self):
        self.count = 0
        from .alc import ActiveLearningClient
        self.act_model = ActiveLearningClient() # parameter in.

    def make_prediction(self):
        self.query_idx, self.preds, self.queried_val = self.act_model.make_preds()
        return (self.query_idx, self.preds, self.queried_val)

    def learn_ratings(self, client_list):
        try:
            self.learned_model, self.X_pool = self.act_model.train_learner(self.query_idx, client_list)
            self.count += 1
            print("LEARN RATINGS COUNT:", self.count)
        except Exception as e:
            print(e)
        return (self.learned_model, self.X_pool)

    def set_x_pool(self):
        self.act_model.get_hearing_group()
        
    def get_prediction(self):
        return (self.query_idx, self.preds, self.queried_val)

    def get_h5(self, file):
        self.file = self.act_model.blob_to_h5(file)
        return self.file
