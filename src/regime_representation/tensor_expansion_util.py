def forward_wrapper(input_states):
    batch_size = input_states.shape[0]
    device = input_states.device

    # Instead of expanding, we should either:
    # 1. Repeat the tensor if we want to replicate the data
    actions_exp = actions.repeat(batch_size // actions.shape[0], 1, 1) if actions.shape[0] != batch_size else actions
    returns_exp = returns_to_go.repeat(batch_size // returns_to_go.shape[0], 1, 1) if returns_to_go.shape[
                                                                                          0] != batch_size else returns_to_go
    timesteps_exp = timesteps.repeat(batch_size // timesteps.shape[0], 1) if timesteps.shape[
                                                                                 0] != batch_size else timesteps

    # Or 2. Slice the input to match the smaller size
    # input_states = input_states[:actions.shape[0]]

    return model(input_states, actions_exp, returns_exp, timesteps_exp)