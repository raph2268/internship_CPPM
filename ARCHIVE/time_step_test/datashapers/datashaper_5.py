import numpy as np
import h5py
import uproot

class DataShaper:
    '''Shapes input h5 or root files to np arrays that can be used for training'''
    def __init__(self, data, target, ofmax, ofcb=None, sig=None):

        self.dig = data
        self.hit = target
        self.ofmax = ofmax
        self.ofcb = ofcb
        self.sig = sig

    @classmethod
    def from_h5(cls, file=None):

        dataset = []

        if file is None:
            file = "/var/home/laatu/dataset/new_datasets/mu140/flat/OF5_rdGap_rdSig/EMB_EMMiddle_0.5125X0.0125_5GeV_OF_WithNoise_000.h5"
        dataset.append(h5py.File(file, 'r'))

        return cls(data=np.concatenate([i['sequence_dig_eT'] for i in dataset]),
            target=np.concatenate([i['sequence_hit_eT'] for i in dataset]),
            ofmax=np.concatenate([i['sequence_OFMax_eT'] for i in dataset]),
            sig=np.concatenate([i['sequence_sig_eT'] for i in dataset])
           )

    @classmethod
    def from_root(cls,
                  root_file=None,
                  data_key='digits_out_sequence_eT',
                  target_key='hit_eT_sequence;1',
                  sig_key='hit_eT_sig_sequence',
                  ofb_key='#tau [#BC]_seq',
                  detector_region='EMB_EMMiddle_0.5125X0.0125;1'):

        if root_file is None:
            root_file = "/var/home/laatu/dataset/new_datasets/mu140/flat/OF5_rdGap_rdSig/digitization_monitorOF5_eta_0.5125_phi_0.0125_EMMiddle_5GeV_WithNoise_000.root"

        with uproot.open(root_file) as events:

            adc = events[detector_region]['0_digitization;1'][data_key].to_numpy()[0]

            target = events[detector_region]['0_digitization;1'][target_key].to_numpy()[0]
            try:
                ofmax = events[detector_region]['2_maxfinder']['digits_out_sequence_eT'].to_numpy()[0]
            except Exception as e:
                ofmax = np.zeros_like(adc)
            try:

                sig = events[detector_region]['0_digitization;1'][sig_key].to_numpy()[0]
            except Exception as e:
                sig = np.zeros_like(adc)

            try:
                ofcb = events[detector_region]['1_of_tau/state/'][ofb_key].to_numpy()[0]
            except Exception as e:
                ofcb = np.zeros_like(ofmax)                

            print("creating from root", np.max(adc), np.max(target))


        return cls(adc, target, ofmax, ofcb, sig)

    def __call__(self, seq_len = 5, sigshift = 5, split=0.5, normalization = (16,16)):


        m_d = normalization[0] # np.max(dig)
        m_t = normalization[1] # np.max(hit)

        dig = self.dig/m_d
        hit = self.hit/m_t

        dig_overlap=np.zeros(shape=(self.dig.shape[0]-seq_len, seq_len),dtype=np.float32)
        hit_overlap=np.zeros(shape=(self.dig.shape[0]-seq_len, seq_len),dtype=np.float32)

        for i in range(seq_len):
            dig_overlap[:,i] = dig[i:self.dig.shape[0]-seq_len+i]
            hit_overlap[:,i] = hit[i:self.dig.shape[0]-seq_len+i]

        data = dig_overlap.reshape(dig_overlap.shape[0], seq_len, 1)
        target = hit_overlap.reshape(dig_overlap.shape[0], seq_len)
        target = target[:,-sigshift]
        target.shape = (dig_overlap.shape[0], 1)

        training_set = 0.45
        v_set = 0.05
        t_set = 1000000

        print("shapes", data.shape, target.shape)

        x_train = data[0:int(data.shape[0]*training_set)-seq_len]
        x_val = data[int(data.shape[0]*training_set):int(data.shape[0]*(training_set+v_set))-seq_len]
        x_test = data[int(data.shape[0]*(training_set+v_set)):]

        y_train = target[0:int(data.shape[0]*training_set)-seq_len]
        y_val = target[int(data.shape[0]*training_set):int(data.shape[0]*(training_set+v_set))-seq_len]
        y_test = target[int(data.shape[0]*(training_set+v_set)):]
        # t_set from 1M


        print("shapes", x_train.shape, x_val.shape, x_test.shape)

        return (x_train, x_val, x_test, y_train, y_val, y_test)

    def get_with_state(self, units=8, hist_size=8, data_size=5, every_nth=3, start_idx=0, end_idx=24):
        """
        Get sliding window with past data in initial hidden state
        Default is vector of 8 with past data from -26 to -5 for hidden states
        data ranges from -1 to +4
        """

        past_vector = self(seq_len=30)
        i_states = past_vector[0][:,start_idx:end_idx:every_nth,:]
        i_states_val = past_vector[1][:,start_idx:end_idx:every_nth,:]
        i_states_test = past_vector[2][:,start_idx:end_idx:every_nth,:]


        if units != hist_size:
            i_states = np.concatenate((i_states, np.zeros((i_states.shape[0], units-hist_size, 1))), axis=1)
            i_states_val = np.concatenate((i_states_val, np.zeros((i_states_val.shape[0], units-hist_size, 1))), axis=1)
            i_states_test = np.concatenate((i_states_test, np.zeros((i_states_test.shape[0], units-hist_size, 1))), axis=1)


        # sliding window data
        pv = []
        for i in range(0,3):
            pv.append(past_vector[i][:,-data_size:,:])

        return ((pv[0], i_states), (pv[1], i_states_val), (pv[2], i_states_test), past_vector[3], past_vector[4], past_vector[5])


def get_data(data):
    '''
    Get numpy arrays from h5
    '''
    ds = DataShaper.from_h5(data)
    X, X_valid, X_test, y, y_valid, y_test = ds()

    return X, X_valid, X_test, y, y_valid, y_test

def get_mixed_data(data, root_data=False, seq_len = 5, sigshift = 5, split=0.5, normalization = (16,16)):
    '''
    Returns numpy arrays with mixed data

    data: array of dataset paths

    '''
    if not root_data:
        shaped = [DataShaper.from_h5(path)(seq_len=seq_len, sigshift=sigshift, split=split, normalization = normalization) for path in data]
    else:
        shaped = [DataShaper.from_root(path)(seq_len=seq_len, sigshift=sigshift, split=split, normalization = normalization) for path in data]

    X = np.concatenate([x[0] for x in shaped])
    X_valid = np.concatenate([x[1] for x in shaped])
    X_test = np.concatenate([x[2] for x in shaped])
    y = np.concatenate([x[3] for x in shaped])
    y_valid = np.concatenate([x[4] for x in shaped])
    y_test = np.concatenate([x[5] for x in shaped])

    return X, X_valid, X_test, y, y_valid, y_test
