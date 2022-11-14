from this import d
import numpy as np
from requests import session
import torch
from subprocess import Popen, STDOUT, PIPE
import pandas as pd
from sklearn.model_selection import train_test_split

from .paths import *
from .graphs import GraphFromCSV, GraphFromTensor

def random_graph_gen(size=50, sample_size=1, states=[-1, 1], to_torch=False, dtype=torch.float, symmetric=True):
    #TODO: add flag for generation of symmetric graphs
    """ Generates sample_size number of random graphs.
    Inputs:
        size: Number of ROIs
        sample_size: Total number of graphs
        states: States from which to choose the values of each node/edge
        to torch: Return the graphs as a tensor
    Outputs:
        graphs: The generated graphs
    """
    graphs = np.random.choice(states, size=(sample_size,size,size))
    if to_torch:
        graphs = torch.tensor(graphs, dtype=dtype)
    return graphs    

def graph_dumper(data_path, graphs, subject_list, suffix='evolved'):
    """ Dumps a tensor with a given number of graphs to a list of csv files.
    Inputs:
        data_path: path to which the csv files will be dumped
        graphs: tensor/array of graphs (Be careful with the dimensions of this object)
        subject_list: List of names that map to the graphs
        suffix: string to add before the extension
    Output: 
        None
    """
    dims = len(graphs.shape)
    for i, sub in enumerate(subject_list):
        name = sub+'_'+suffix+'.csv'
        if dims == 1:
            gr = graphs.numpy().reshape((1, graphs.shape[-1]))
        else:
            gr = graphs[i].numpy().reshape((1,graphs.shape[-1]))
        dataframe = pd.DataFrame(data=gr.astype(float))
        dataframe.to_csv(data_path+name, sep=',', header=False, float_format='%.6f', index=False)

def common_subjects(data_path, sessions, subject_ID='*', format=".csv"):
    """ Finds the common subjects between two sessions """
    # Loading files
    files = list()
    for ses in sessions:
        files.extend(get_subjects(data_path, ses, subject_ID, format=format))
    # Creating lists
    subjects_pre = set()
    subjects_post = list()
    for f in files:
        _, session, patient, _ = get_info(f)
        if 'preop' in session:
            subjects_pre.add(patient)
        else:
            subjects_post.append(patient)
    # Finding intersection
    return subjects_pre.intersection(subjects_post), subjects_pre.symmetric_difference(subjects_post)

def two_session_graph_loader(data_path, rois=170, augmentation=1, mu=0, sigma=1, sessions=['preop', 'postop'], subject_ID='*', norm=False):
    """
    Loads data that is available in two sessions. Training data is augmented as many times as 'augmentation' and lognormal noise is added.
    """
    if augmentation == 1:
        noise = False
    else:
        noise = True

    features = int(rois*(rois-1)/2)
    subjects, _ = common_subjects(data_path, sessions=sessions, subject_ID=subject_ID)
    tr_samples = len(subjects)
    pre_cond, post_cond = np.zeros((tr_samples*augmentation, features)), np.zeros((tr_samples*augmentation, features))
    ordered_subjects = list()

    noise_gen = np.random.default_rng()
    for i, pat in enumerate(subjects):
        ordered_subjects.append(pat)
        output = Popen(f"find {data_path if len(data_path)>0 else '.'} -wholename *{pat}*.csv", shell=True, stdout=PIPE)
        files = str(output.stdout.read()).strip("b'").split('\\n')[:-1]
        if 'preop' in files[0]:
            preop_graph = pd.read_csv(files[0], delimiter=',', header=None).values[0, :features]
            postop_graph = pd.read_csv(files[1], delimiter=',', header=None).values[0, :features]
        else:
            preop_graph = pd.read_csv(files[1], delimiter=',', header=None).values[0, :features]
            postop_graph = pd.read_csv(files[0], delimiter=',', header=None).values[0, :features]
       
        for aug in range(augmentation):
            pre_cond[i+aug*tr_samples,:] = np.log1p(preop_graph) + noise*noise_gen.lognormal(mu, sigma, size=preop_graph.shape)
            post_cond[i+aug*tr_samples,:] = np.log1p(postop_graph) + noise*noise_gen.lognormal(mu, sigma, size=postop_graph.shape)

            if norm:
                pre_cond[i+aug*tr_samples,:] = pre_cond[i+aug*tr_samples,:]/np.max(pre_cond[i+aug*tr_samples,:])
                post_cond[i+aug*tr_samples,:] = post_cond[i+aug*tr_samples,:]/np.max(post_cond[i+aug*tr_samples,:])
    
    return (torch.tensor(pre_cond, dtype=torch.float64), torch.tensor(post_cond, dtype=torch.float64)), ordered_subjects

def graph_loader(data_path, unflat, session='*', subject_ID='*', rois=170, augmentation=1, mu=0, sigma=1, norm=False):
    """ 
    Loads data from specific sessions and/or subjects. Returns numpy array of flattened graphs.
    Need to provide wether the graphs should be unlfattened or not.
    """
    # Data augmentation through noise
    if augmentation == 1:
        noise = False
    else:
        noise = True
    noise_gen = np.random.default_rng()

    files = get_subjects(data_path, session, subject_ID)
    subjects = list()
    if unflat:
        # 3D array
        graphs = np.zeros((len(files)*augmentation, rois, rois))
    else:
        # Number of features - it makes sense for unprocessed (unshufled) graphs
        features = int(rois*(rois-1)/2)
        # 2D array
        graphs = np.zeros((len(files)*augmentation, features))

    for i, f in enumerate(files):
        path, session, subject, name = get_info(f)
        subjects.append([path, session, subject, name])
        for aug in range(augmentation):
            # Get connections
            if unflat:
                graphs[i+aug*len(files),:,:] = GraphFromCSV(f, name, rois=rois).unflatten_graph(to_default=False, save_flat=False)
            else:
                graphs[i+aug*len(files),:] = GraphFromCSV(f, name, rois=rois).get_connections()
            # Log(1+x) and add noise
            graphs[i+aug*len(files)] = np.log1p(graphs[i+aug*len(files)]) + noise*noise_gen.lognormal(mu, sigma, size=graphs[i+aug*len(files)].shape)
            # Normalize
            if norm:
                graphs[i+aug*len(files)] = graphs[i+aug*len(files)]/np.max(graphs[i+aug*len(files)])
    return graphs, np.array(subjects)            

def subset(data, cut=1):
    """ Returns a subset of the data """
    return data[:1]

def unflatten_data(graph, rois, dtype=torch.float, norm=True):
    N = graph.shape[0]
    UN = np.zeros((N, rois, rois))
    for i in range(N):
        UN[i] = GraphFromTensor(graph[i].unsqueeze(0), '', rois=rois).unflatten_graph(to_default=False, save_flat=False)
        UN[i] = UN[i]/np.max(UN[i]) if norm else UN[i]
    return torch.tensor(UN, dtype=dtype)

def flatten_graph(graphs, dtype=torch.float, norm=True):
    """
    Flatten the lower triangular adjancency matrix of the graph. 
    The flattened graph becomes available after applying this method.
    """
    N = graphs.shape[0]
    dims = int(graphs[0].shape[0]*(graphs[0].shape[0]-1)/2)
    flat_conns = np.zeros((N,dims))
    for n in range(N):
        graph = graphs[n]
        k = 0
        for i in range(graph.shape[0]):
            for j in range(i):
                flat_conns[n,k] = graph[i,j]
                k += 1
    return torch.tensor(flat_conns, dtype=dtype)

def delete_rois(graphs, ROIs):
    """ Deletes from the graphs certain amount of ROIs from the numpy adjacency matrices. 
    Inputs:
        graphs: Numpy array of N graphs, each one with dimensions ROIs x ROIs
        ROIs: Array of regions to delete
    Returns:
        cut_graphs: Numpy array of cutted graphs
        rois_dict: dictionary containing the kept ROIs and the deleted ones. Also the new reordering
    """
    new_rois, k, ROIs = dict(), 1, np.array(ROIs)
    for r in range(1, graphs.shape[1]+1):
        if r not in ROIs-1:
            new_rois[k] = r
            k += 1
    graphs = np.delete(graphs, obj=ROIs-1, axis=1)
    graphs = np.delete(graphs, obj=ROIs-1, axis=2)

    return graphs, new_rois

def prepare_data_healthy_pre_post(path, rois, norm=False, flatten=True, del_rois=None, augmentation=1, mu=0, sigma=1, dtype=torch.float64):
    """
    Highly specific function that prepares the data for the model.
    Returns CONTROLS and PATIENTS in unflattened graphs.
    """

    # Preparing CONTROL
    _, unique_CON = common_subjects(path, subject_ID="CON")
    (pre_CON, post_CON), common_CON = two_session_graph_loader(path, subject_ID="CON", augmentation=augmentation, mu=mu, sigma=sigma, norm=norm)
    CONTROL, CON_subjects = 0.5*(pre_CON+post_CON), common_CON # Mean of the paired control subjects
    for s in unique_CON:
        x, _ = graph_loader(path, unflat=False, subject_ID=s, augmentation=augmentation, mu=mu, sigma=sigma, norm=norm)
        CONTROL = torch.cat((CONTROL, torch.tensor(x, dtype=dtype)), dim=0)
        CON_subjects.append(s)

    # Preparing preop-postop PATIENTS
    _, unique_PAT = common_subjects(path, subject_ID="PAT")
    PATIENT_2sessions, PAT_subjects = two_session_graph_loader(path, subject_ID="PAT", augmentation=augmentation, mu=mu, sigma=sigma, norm=norm)

    # Preparing unmatched preop PATIENTS
    PATIENT_1session = torch.zeros((len(unique_PAT),PATIENT_2sessions[0].shape[1]))
    for i, s in enumerate(unique_PAT):
        x, _ = graph_loader(path, unflat=False, session="preop",subject_ID=s)
        PATIENT_1session[i, :] = torch.tensor(x, dtype=dtype)
        PAT_subjects.append(s)

    CONTROL = unflatten_data(CONTROL, rois, norm=norm)
    pre, post = unflatten_data(PATIENT_2sessions[0], rois, norm=norm), unflatten_data(PATIENT_2sessions[1], rois, norm=norm)
    PATIENT_1session = unflatten_data(PATIENT_1session, rois, norm=norm)

    if del_rois is not None:
        CONTROL, rois = delete_rois(CONTROL, ROIs=del_rois)
        pre, _ = delete_rois(pre, ROIs=del_rois)
        post, _ = delete_rois(post, ROIs=del_rois)
        PATIENT_1session,_ = delete_rois(PATIENT_1session, ROIs=del_rois)

    if flatten:
        CONTROL = flatten_graph(CONTROL, norm=norm)
        pre, post = flatten_graph(pre, norm=norm), flatten_graph(post, norm=norm)
        PATIENT_1session = flatten_graph(PATIENT_1session, norm=norm)

    return (CONTROL, CON_subjects), ((pre, post), PAT_subjects[0:len(PAT_subjects)-i-1]), (PATIENT_1session, PAT_subjects[len(PAT_subjects)-i-1:])

def data_splitter(data, split=[70, 20, 10]):
    """
    Splits the data into train, test set. Optional to obtain a validation set.
    Inputs:
        data: list of graphs
        split: list of percentatges (train, val, test). Has to add to 100.
    Returns:
        train_data: List of training graphs
        val_data: List of validating graphs (optional)
        test_data: List of testing graphs
    """

    x = range(len(data))
    tr, ts = train_test_split(x, test_size=split[-1]*0.01)
    if len(split)==3:
        tr, val = train_test_split(tr, test_size=split[1]*0.01*len(data)/len(tr))
        return [data[i] for i in tr], [data[j] for j in val], [data[k] for k in ts]
    else:
        return [data[i] for i in tr], [data[k] for k in ts]

def prepare_functional_files(data_path, sessions, subject, exclude=''):
    """
    Returns the nifti files in a useful format.
    Inputs:
        data_path: path to the data
    """
    
    paired, unpaired = common_subjects(data_path, sessions=sessions, subject_ID=subject, format=".nii.gz")
    data_paired, data_unpaired = [], []
    for s in paired:
        data_paired.append([
            get_subjects(data_path, session='preop', subject_ID=s, format='.nii.gz', exclude=exclude)[0],
            get_subjects(data_path, session='postop', subject_ID=s, format='.nii.gz', exclude=exclude)[0]
            ])
    for s in unpaired:
        data_unpaired.append(get_subjects(data_path, subject_ID=s, format='.nii.gz', exclude=exclude)[0])
    return (data_paired, list(paired)), (data_unpaired, list(unpaired))

def prepare_structural_healthy(path, sessions, rois, norm=False, flatten=True, del_rois=None, augmentation=1, mu=0, sigma=1, dtype=torch.float64):
    _, unique_CON = common_subjects(path, sessions=sessions, subject_ID="CON")
    (pre_CON, post_CON), common_CON = two_session_graph_loader(path, subject_ID="CON", augmentation=augmentation, mu=mu, sigma=sigma, norm=norm)
    CONTROL, CON_subjects = 0.5*(pre_CON+post_CON), common_CON # Mean of the paired control subjects
    for s in unique_CON:
        x, _ = graph_loader(path, unflat=False, subject_ID=s, augmentation=augmentation, mu=mu, sigma=sigma, norm=norm)
        CONTROL = torch.cat((CONTROL, torch.tensor(x, dtype=dtype)), dim=0)
        CON_subjects.append(s)
    CONTROL = unflatten_data(CONTROL, rois, norm=norm)
    if del_rois is not None:
        CONTROL, _ = delete_rois(CONTROL, ROIs=del_rois)
    if flatten:
        CONTROL = flatten_graph(CONTROL, norm=norm)        
    return CONTROL, CON_subjects

def prepare_structural_methods(data_path, session, rois, norm=False, flatten=True, del_rois=None, augmentation=1, mu=0, sigma=1, dtype=torch.float64):
    files = get_subjects(data_path, session=session, subject_ID="PAT", format='.csv')
    features = rois*(rois-1)//2

    graphs = torch.zeros((len(files),features))
    subjects = []
    for i, file in enumerate(files):
        _, _, subject, _ = get_info(file)
        subjects.append(subject)
        x, _ = graph_loader(data_path, unflat=False, session=session, subject_ID=subject, augmentation=augmentation, mu=mu, sigma=sigma, norm=norm)
        graphs[i, :] = torch.tensor(x, dtype=dtype)
    graphs = unflatten_data(graphs, rois, norm=norm)
    if del_rois is not None:
        graphs, rois = delete_rois(graphs, ROIs=del_rois)
    if flatten:
        graphs = flatten_graph(graphs, norm=norm)  
    return graphs, subjects

def match_structural_methods(method1, subjects1, method2, subjects2):
    matched2 = torch.zeros(method1.shape)
    for i, s2 in enumerate(subjects2):
        for j, s1 in enumerate(subjects1):
            if s2==s1:
                matched2[j] = method2[i]
    return method1, matched2, subjects1

if __name__ == '__main__':
    
    pass