from tensorboard.plugins.hparams import api as hp

HP_MODEL = hp.HParam("model")
HP_FEATURE_MAPS = hp.HParam("feature_maps", display_name="Feature maps")
HP_QUERY_MAPS = hp.HParam("query_maps", display_name="Query maps")
HP_TRAIN_ROUNDS = hp.HParam("train_rounds", display_name="Train rounds")
HP_TEST_ROUNDS = hp.HParam("test_rounds", display_name="Train rounds")
HP_MLP_LAYERS = hp.HParam("mlp_layers", display_name="Layers in MLP")
HP_TRAINABLE_PARAMS = hp.HParam("trainable_params", display_name="Trainable parameters")
HP_TASK = hp.HParam("task", display_name="Task")
