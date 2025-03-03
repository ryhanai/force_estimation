def normalization(data, indataRange, outdataRange=[0.1, 0.9]):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data


def tensor2numpy(x):
    """
    Convert tensor to numpy array.
    """
    if x.device.type == 'cpu':
        return x.detach().numpy()
    else:
        return x.cpu().detach().numpy()

    
OK = '\033[92m'
WARN = '\033[93m'
NG = '\033[91m'
END_CODE = '\033[0m'

def print_info(msg):
    print( OK + "[INFO] " + END_CODE + msg )

def print_warn(msg):
    print( WARN + "[WARNING] " + END_CODE +  msg )

def print_error(msg):
    print( NG + "[ERROR] " + END_CODE + msg )
