def print_parameters(parameters):
    '''
    parameters = {
        "limit": limit,
        "n_tiles": num_tiles,
        "search_rotations": search_rotations,
        "search_symmetry": search_symmetry,
        "upsize_depth_search": upsize_depth_search,
        "quality": quality,
        'strategy': strategy_name,
        'sample': sample_network,
        'sample_temp': sample_temperature,
    }'''

    print('Generating a mosaic with the following parameters:')
    print(' - Number of tiles:', parameters['n_tiles'])
    print(' - Size of the dataset search limit:', parameters['limit'])
    print(' - Searching rotations:', parameters['search_rotations'])
    print(' - Searching symmetries:', parameters['search_symmetry'])
    print(f' - Searching upsizing up to {parameters["upsize_depth_search"]}')
    print(' - Full quality generation:', parameters['quality'])
    print(' - Strategy chosen:', parameters['strategy'])
    if parameters['strategy'] == 'NN':
        print(' - Using sampling for the Network probabilities:', parameters['sample'])
        if parameters['sample']:
            print(' - Sampling temperature:', parameters['sample_temp'])

