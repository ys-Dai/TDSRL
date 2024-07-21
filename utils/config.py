import os

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = 'dataset/processed/'
LOG_DIR = 'logs/'
DATA_PROPERTY_DIR = 'data/'


DATASET_LIST = ['SMAP', 'SMD', 'SWaT', 'NeurIPS']

TRAIN_DATASET = {}
TEST_DATASET = {}
TEST_LABEL = {}

for data_name in DATASET_LIST:
    TRAIN_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_train.npy')
    TEST_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_test.npy')
    TEST_LABEL[data_name] = os.path.join(DATASET_DIR, data_name + '_test_label.npy')


DATA_DIVISION = {'SMAP' : {'channel' : DATA_PROPERTY_DIR+'SMAP_test_channel.json',
                           'class' : DATA_PROPERTY_DIR+'SMAP_test_class.json'},
                 'SMD' : {'channel' : DATA_PROPERTY_DIR+'SMD_test_channel.json'},
                }

DEFAULT_DIVISION = {'SMAP' : 'channel',
                    'SMD' : 'channel',
                    'SWaT' : 'total',
                    'NeurIPS' : 'total',
                   }

NUMERICAL_COLUMNS = {'SMAP' : (0,),
                     'SMD' : tuple(list(range(7)) + list(range(8, 38))),
                     'NeurIPS' : (0,)
                    }

CATEGORICAL_COLUMNS = {'SMAP' : range(1, 25),
                       'SMD' : (7,),
                       'SWaT' : tuple([2,3,4,9] + list(range(11, 16)) + list(range(19, 25))\
                                      + list(range(29, 34)) + [42,43,48,49,50]),
                        'NeurIPS' : (0,)
                      }

IGNORED_COLUMNS = {'SWaT' : (10,)
                  }


# SMD series
train_smd = {'SMD{}'.format(i) : os.path.join(DATASET_DIR, 'SMD{}_train.npy'.format(i)) for i in range(28)}
test_smd = {'SMD{}'.format(i) : os.path.join(DATASET_DIR, 'SMD{}_test.npy'.format(i)) for i in range(28)}
label_smd = {'SMD{}'.format(i) : os.path.join(DATASET_DIR, 'SMD{}_test_label.npy'.format(i)) for i in range(28)}
division_smd = {'SMD{}'.format(i) : 'total' for i in range(28)}
numerical_smd = {'SMD{}'.format(i) : NUMERICAL_COLUMNS['SMD'] for i in range(28)}
categorical_smd = {'SMD{}'.format(i) : (7,) for i in range(28)}

TRAIN_DATASET.update(train_smd)
TEST_DATASET.update(test_smd)
TEST_LABEL.update(label_smd)
DEFAULT_DIVISION.update(division_smd)
NUMERICAL_COLUMNS.update(numerical_smd)
CATEGORICAL_COLUMNS.update(categorical_smd)


NUMERICAL_COLUMNS['SWaT'] = tuple([i for i in range(0, 51) if (i not in CATEGORICAL_COLUMNS['SWaT'])\
                                   and (i not in IGNORED_COLUMNS['SWaT'])])