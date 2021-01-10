import tabnet
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

def TabNetModel(n_inputs=146, n_classes=111, n_latent_features=12, output_dim=8, len_dataset=743):
    model = tabnet.TabNetClassifier(
        None, num_classes=n_classes, feature_dim=n_latent_features,
        num_features=n_inputs, output_dim=output_dim,
        num_decision_steps=2, relaxation_factor=1.3,
        sparsity_coefficient=1e-3, batch_momentum=0.98,
        virtual_batch_size=None, norm_type='group',
        num_groups=1
    )

    lr = ExponentialDecay(
        0.00003,
        decay_steps=len_dataset,
        decay_rate=0.95,
        staircase=False
    )

    optimizer = Adam(lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model