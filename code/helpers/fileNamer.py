# a helper function which creates a constant format of file names
def fileNamer(node_model=None, dataset_name=None, model_name=None, l_inf=None, num_layers=None, seed=None,
              targeted=None, attack_epochs=None, patience=None, start=None, end=None):
    file_name = ''

    if node_model is not None:
        if node_model is True:
            node_model = 'NodeModel'
        else:
            node_model = 'EdgeModel'
    l_inf = 'Linf' + str(l_inf) if l_inf is not None else l_inf
    num_layers = str(num_layers) + 'Layers' if num_layers is not None else num_layers
    seed = 'Seed' + str(seed) if seed is not None else seed
    if targeted is not None:
        targeted_attack_str = ''
        if not targeted:
            targeted_attack_str += 'un'
        targeted = targeted_attack_str + 'targeted'
    attack_epochs = str(attack_epochs) + 'K' if attack_epochs is not None else attack_epochs
    patience = 'patience' + str(patience) if patience is not None else patience

    for input in [start, node_model, dataset_name, model_name, num_layers, seed, l_inf, targeted, attack_epochs,
                  patience]:
        if input is not None:
            file_name += '_' + input

    if end is not None:
        file_name += end
    return file_name[1:]
