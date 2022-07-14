import pandas as pd

from dataExtract.data_process import data_pro as c_data_pro
from config.config import get_cfg
import tools.model_op as m_op
import joblib


def split_df_by_col(df_, label_):
    ft = df_.groupby(label_)
    listType = df_[label_].unique()
    listDf = {}
    for name in listType:
        curDf = ft.get_group(name)  #.reset_index()
        listDf[name] = curDf
    return listDf

def load_config(mode, basePath, filename, label):
    model_file_cfg = "E:/project/vispek/spec_algo/dev/spec_algo/config/param_cfg/" + mode + ".yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_file_cfg)
    cfg.MODEL.MODEL_NAME = mode
    cfg.MODEL.SAVE_PATH = basePath + filename + label
    cfg.TEST.SAVE_PLOT_PATH = basePath + 'plot/' + filename + label
    cfg.TEST.SAVE_PLOT = False

    cfg.TRAIN.ITER = 300
    cfg.TRAIN.TOL = 0.1
    return cfg

def train_person_data(val, key, feature, label, cfg, basePath, filename, ft_list
                      , estimate_value, truth_value, mode, TRAIN = True, test_val = None):
    dataEx_train = c_data_pro('file', feature_name=feature, label_name=label)
    dataEx_train.feature_matrix = val.iloc[:, 0:44]
    dataEx_train.label_matrix = val.iloc[:, -1:]
    dataEx_pre = dataEx_train
    if test_val is not None:
        dataEx_pre = c_data_pro('file', feature_name=feature, label_name=label)
        dataEx_pre.feature_matrix = test_val.iloc[:, 0:44]
        dataEx_pre.label_matrix = test_val.iloc[:, -1:]

    cfg.TEST.SAVE_PLOT_PATH = basePath + 'plot/' +  filename + label + '_' + str(key)
    cfg.MODEL.SAVE_PATH = basePath + filename + label + '_' + str(key)
    if TRAIN:
        model, acc = m_op.cls_model_train(dataEx_train, cfg, step=0, data_val_EX = dataEx_pre
                                          , ft_list=ft_list
                                          )
    else:
        model = joblib.load(cfg.MODEL.SAVE_PATH + '_'+ mode + '.pkl')

    y_predict, acc = m_op.reg_model_test(dataEx_pre, model, cfg
                                         , ft_list=ft_list
                                         )
    estimate_value.append(y_predict)
    truth_value.append(dataEx_pre.label_matrix.values)

    return estimate_value, truth_value



basePath = "E:/data/bloodsugar/bloodsugar2/data/"

if __name__ == '__main__':
    label = '血糖值'
    feature = '光谱数据'
    mode = "LASSO"

    filename = "bloodsugar_sel_12_lined"

    # selected features
    ft_list = [34,35,36, 37,38,39]

    ID_predict = 13

    # load data
    data_train = pd.read_excel(basePath + "bloodsugar_sel_12_lined.xls")
    data_pre = pd.read_excel(basePath + "bloodsugar_sel_12_lined_pre.xls")

    # split every body's data
    listDf = split_df_by_col(data_train, '姓名')
    listType = data_train['姓名'].unique()

    # train every body's data
    cfg = load_config(mode, basePath, filename, label)

    truth_value = []
    estimate_value = []
    for key in listType:
        print(key)
        val = listDf[key]
        estimate_value, truth_value = train_person_data(val, key, feature, label, cfg, basePath, filename, ft_list
                      , estimate_value, truth_value, mode, TRAIN = True)

    # train all people data
    estimate_value, truth_value = train_person_data(data_train, 'all', feature, label, cfg, basePath, filename, ft_list
                  , estimate_value, truth_value, mode, TRAIN = True)


    # Forecast for one of them
    val = listDf[13]
    estimate_value_13 = []
    truth_value_13 = []
    estimate_value_13, truth_value_13 = train_person_data(val, 13, feature, label, cfg, basePath, filename, ft_list
                                                    , estimate_value_13, truth_value_13, mode, TRAIN=False,
                                                    test_val=data_pre)