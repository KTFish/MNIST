import wandb
from grid_search_train import train


# Let's choose some hyperparameters to test
parameters = {
    "batch_size": {"values": [32]},
    "epochs": {"values": [5, 10, 2]},
    "lr": {"values": [0.001, 0.01]},
    "hidden_units": {"values": [5, 10, 15]},
    # Convolution hyperparameters
    "kernel_size": {"values": [3]},  # Size of the filter
    "padding": {"values": [0, 1, 2]},  # 'Bonus' pixels added around the image
    "stride": {"values": [1, 2]},  # Number of pixels we move the filter at a time
}

# Now we create a `sweep_config` dict
sweep_config = {
    "method": "random",  # The parameters will be picked up randomly
    "parameters": parameters,
    "name": "MNIST Sweep",
    "project": "MNIST-Czarna-Magia",  # Name of our project
    "metric": {
        "name": "accuracy",
        "goal": "maximize",
    },  # The higher the accuracy the better
    "description": "Searching for the best hyperparameters...",
}

# Initialize this sweep, this will output a link to the visualisation
sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])

# Run the trainign script as an agent to start the hyperparameter sweeping process
NUM_OF_TESTS = 1
wandb.agent(sweep_id, function=train, count=NUM_OF_TESTS)

# kernel_size = 3
# padding = 0
# stride = 1
# traverse = lambda n: (n - kernel_size + 2 * padding) / stride + 1
# n = int(traverse(traverse(28)))
