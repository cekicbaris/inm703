from model import *

# -------------------
# Blocking Design 
# -------------------
# Define the model
experiment = Model_Rescorla_Wagner(experiment_name="Blocking", lambda_US=1, beta_US=0.5)

# Define the predictors
A = Predictor(name='A', alpha = 0.2)
B = Predictor(name='B',alpha = 0.2)
C = Predictor(name='C',alpha = 0.2)

# Define the experiment groups
experiment_group = Group(name="Experiment Group")
experiment_group.add_phase_for_group(phase_name='Conditioning', predictors=[A], outcome=True, number_of_trial=10)
experiment_group.add_phase_for_group(phase_name='Blocking', predictors=[A], outcome=True, number_of_trial=10)
experiment.add_group(experiment_group)

control_group = Group(name="Control Group")
control_group.add_phase_for_group(phase_name='Conditioning', predictors=[C], outcome=True, number_of_trial=10)
control_group.add_phase_for_group(phase_name='Blocking', predictors=[A,B], outcome=True, number_of_trial=10)
experiment.add_group(control_group)

# Run the model
experiment.model_run()
experiment.display_results(save_to_file=True)