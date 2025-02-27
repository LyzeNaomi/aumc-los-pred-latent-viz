#import dependencies

from ast import Raise
from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.progress_reporter import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback

from train import train


class hyper_params():
    def __init__(self, config):
        self.config = config

    def main_hyperparam(self, num_samples=10, max_num_epochs=20, tune_scheduler=None, wandb=True):
        if tune_scheduler == 'ASHA':
            if wandb:
                project_n = self.config['model_type'] + '_asha-schedulers'
                callbacks=[
                WandbLoggerCallback(api_key=self.config['api_key'], project=project_n)
                ]
            elif not wandb:
                callbacks = None

            scheduler = ASHAScheduler(
            metric = "val_loss",
            mode = "min",
            max_t = max_num_epochs,
            grace_period = 1, 
            reduction_factor = 2)
        
        elif tune_scheduler == 'PBT':
            print('Using PBT')
            if wandb:
                project_n = self.config['model_type'] + '_pbt-schedulers'
                callbacks=[
                WandbLoggerCallback(api_key=self.config['api_key'], project=project_n)
                ]
            elif not wandb:
                callbacks = None

            scheduler = PopulationBasedTraining(
                time_attr='training_iteration',
                metric='loss',
                mode='min',
                perturbation_interval=2,
                #This needs to be more dynamically defined based on what is in the config file
                # hyperparam_mutations=self.config['pbt_dict']
                hyperparam_mutations={
                        #"num_layers": tune.choice([1, 2, 3, 4]),
                        "dropout_prob":tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]), 
                        "learning_rate":tune.loguniform(1e-4, 1e-1),
                        "bsize": tune.choice([32, 64, 128, 256, 512, 1024])
                }
                )

        else:
            raise Exception('Improper specification of the scheduler')


        reporter = CLIReporter(
        metric_columns = ["loss", "val_loss", "training_iteration"])
        

        self.result = tune.run(
        partial(train, checkpoint_dir=self.config['path']),
        config = self.config,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter,
        callbacks=callbacks)
        
        self.best_trial = self.result.get_best_trial("loss", "min", "last")
        print(f"best_trial: {self.best_trial}")
        '''
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))
        
        best_trained_model = GRUModel(config['input_dim'], best_trial.config['hidden_dim'], best_trial.config['num_layers'], 
                                                    config['output_dim'], best_trial.config['dropout_prob'], attention = False) #now fit the model with the best hyps | how do I do to return the parms of the train fux up
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
                    best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        
        #test_acc = test_accuracy(best_trained_model, device)
        #print("Best trial test set accuracy: {}".format(test_acc))
        '''
        return self.best_trial, self.result

    # if __name__ == "__main__":
    #     best_trial, result = main_hyperparam(num_samples=2, max_num_epochs=10)