# a helper function which creates a constant format of file names
def fileNamer(node_model: str = None, dataset_name: str = None, model_name: str = None, l_inf: float = None,
              l_0: float = None, num_layers: int = None,
              seed: int = None, targeted: bool = None, continuous_epochs: int = None, patience: int = None,
              start: str = None, end: str = None) -> str:
    """
        creates the generic name of the output file

        Parameters
        ----------
        node_model: str - node or edge model
        dataset_name: str
        model_name: str
        l_inf: float -
        l_0: float
        num_layers: int
        seed: int
        targeted: bool
        continuous_epochs: int
        patience: int
        start: str - prefix for the file name
        end: str - suffix for the file name

        Returns
        -------
        file_name: str
    """

    file_name = ''

    if node_model is not None:
        if node_model is True:
            node_model = 'NodeModel'
        else:
            node_model = 'EdgeModel'
    l_inf = 'Linf' + str(l_inf) if l_inf is not None else l_inf
    l_0 = 'AttrRatio' + str(l_0) if l_0 is not None else l_0
    num_layers = str(num_layers) + 'Layers' if num_layers is not None else num_layers
    seed = 'Seed' + str(seed) if seed is not None else seed
    if targeted is not None:
        targeted_attack_str = ''
        if not targeted:
            targeted_attack_str += 'un'
        targeted = targeted_attack_str + 'targeted'
    continuous_epochs = str(continuous_epochs) + 'K' if continuous_epochs is not None else continuous_epochs
    patience = 'patience' + str(patience) if patience is not None else patience

    for input in [start, node_model, dataset_name, model_name, num_layers, seed, l_inf, l_0, targeted,
                  continuous_epochs, patience]:
        if input is not None:
            file_name += '_' + input

    if end is not None:
        file_name += end
    return file_name[1:]
