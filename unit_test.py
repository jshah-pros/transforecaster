import numpy as np
from model.transforecaster import Transforecaster

# Create dummy data
encoder_inputs = np.random.randn(10, 512, 12)
decoder_inputs = np.random.randn(10, 24, 1)
decoder_outputs = np.random.randn(10, 24, 1)


# Model Parameters
num_layers = 6
num_heads = 7
dff = 2048
d_model = 4
n_features = encoder_inputs.shape[-1]
n_steps_out = decoder_outputs.shape[1]
dropout_rate = 0.2

# Create transforecaster
transforecaster = Transforecaster(
    num_layers=num_layers,
    num_heads=num_heads,
    dff=dff,
    d_model=d_model,
    n_features=n_features,
    n_steps_out=n_steps_out,
    dropout_rate=dropout_rate
)

# Run a forward pass
output = transforecaster((encoder_inputs, decoder_inputs), training = False)
print(output.shape) # Must equal decoder_outputs.shape