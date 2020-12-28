from model import *
# -------------------
# Extinction Design
# -------------------

# Define the model
exitinction = Model_Rescorla_Wagner(experiment_name="Extinction", lambda_US=1, beta_US=0.5)

# Define the predictors
A = Predictor(name='A', alpha = 0.2)
# Define the experiment groups
exitinction_group = Group(name="Experiment Group")
exitinction_group.add_phase_for_group(phase_name='Conditioning', predictors=[A], outcome=True, number_of_trial=10)
exitinction_group.add_phase_for_group(phase_name='Extinction', predictors=[A], outcome=False, number_of_trial=10)
exitinction.add_group(exitinction_group)

# Run the model
exitinction.model_run()
exitinction.display_results(save_to_file=True)