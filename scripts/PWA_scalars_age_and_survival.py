import sys
from PWA_Class import Predictions

# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Survival')  # target
    sys.argv.append('all+PWA')  # predictors
    sys.argv.append('CNN')  # algo_name

# Compute results
self = Predictions(target=sys.argv[1], predictors=sys.argv[2], algo_name=sys.argv[3])
self.preprocessing()
self.hyperparameters_tuning()
self.train_model()
self.evaluate_model_performance()
self.generate_feature_importance()
self.save_predictions()

# Exit
print('Done.')
sys.exit(0)
