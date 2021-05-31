from tensorflow.python.keras.models import load_model
from settings import CURRENT_DATA_VERSION
from src.features.helpers.data_load import get_train_source, get_model_name, get_train_val_test_data, get_model_type, \
    get_img_train_val_test_data
from src.features.helpers.metrics import get_f1
from src.features.helpers.training import plot_prediction_results, get_model_path, plot_confusion_matrix

ML_MODEL = get_model_name()

model_type = get_model_type()

train_source = get_train_source()

if CURRENT_DATA_VERSION == 'v4':
    x_train, y_train, x_val, y_val, x_test, y_test = get_img_train_val_test_data()
else:
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_val_test_data(train_source, version=CURRENT_DATA_VERSION)

model = load_model(
    get_model_path(model_type, ML_MODEL),
    custom_objects={'f1_score': get_f1()}
)
y_pred_train_classes = (model.predict(x_train) > 0.5).astype("int32")
plot_confusion_matrix(y_pred_train_classes, y_train, [0, 1], 'Training', ML_MODEL, model_type)

y_pred_val_classes = (model.predict(x_val) > 0.5).astype("int32")
plot_confusion_matrix(y_pred_val_classes, y_val, [0, 1], 'Validation', ML_MODEL, model_type)

y_pred_test_classes = (model.predict(x_test) > 0.5).astype("int32")
plot_confusion_matrix(y_pred_test_classes, y_test, [0, 1], 'Test', ML_MODEL, model_type)

plot_prediction_results(
    y_pred_train_classes,
    y_train,
    y_pred_val_classes,
    y_val,
    y_pred_test_classes,
    y_test,
    ML_MODEL,
    model_type
)
