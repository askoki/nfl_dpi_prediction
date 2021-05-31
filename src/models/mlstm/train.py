import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from src.features.helpers.data_load import get_train_val_test_data, get_model_name, get_train_source, \
    get_and_print_distribution, get_model_args
from src.features.helpers.metrics import get_model_prediction_results
from src.features.helpers.training import plot_training_validation_loss, get_save_best_model, \
    get_early_stopping_definition
from src.models.mlstm.models import generate_model_mlstm
from settings import Models, CURRENT_DATA_VERSION, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE

# -------- required arguments --------

ML_MODEL = get_model_name()

minority_class_division_factor, num_layers, num_cells, monitor = get_model_args()

train_source = get_train_source()

# ---------------- END OF ARGUMENTS --------------------


x_train, y_train, x_val, y_val, x_test, y_test = get_train_val_test_data(
    train_source,
    version=CURRENT_DATA_VERSION
)

num_negatives, num_positives = get_and_print_distribution(y_train)

classes = np.unique(y_train)
le = LabelEncoder()
y_ind = le.fit_transform(y_train.ravel())
recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
class_weight = recip_freq[le.transform(classes)]
class_weight = {0: class_weight[0], 1: class_weight[1]}
print("Class weights : ", class_weight)

EPOCHS = DEFAULT_EPOCHS
BATCH_SIZE = DEFAULT_BATCH_SIZE
model = KerasClassifier(
    build_fn=generate_model_mlstm,
    input_dim=(x_train.shape[1], x_train.shape[2]),
    cells=num_cells,
    learning_rate=1e-4,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f'{datetime.now().strftime("%H:%M:%S")} Start training...')

model_checkpoint = get_save_best_model(Models.MLSTM.value, ML_MODEL, monitor=monitor)
factor = 1. / np.cbrt(2)
reduce_lr = ReduceLROnPlateau(
    monitor=monitor,
    patience=DEFAULT_EPOCHS / 2,
    mode='max',
    factor=factor,
    cooldown=0,
    min_lr=1e-5,
    verbose=2
)
early_stopping = get_early_stopping_definition(
    monitor=monitor,
    mode='max',
    patience=DEFAULT_EPOCHS / 4
)
callback_list = [model_checkpoint, reduce_lr, early_stopping]
history = model.fit(
    x_train,
    y_train,
    class_weight=class_weight,
    validation_data=(x_val, y_val),
    callbacks=callback_list
)

print(f'{datetime.now().strftime("%H:%M:%S")} End training...')
print("Saving model...")
model.model.summary()

plot_training_validation_loss(history, ML_MODEL, Models.MLSTM.value)

y_pred_train_classes = (model.model.predict(x_train) > 0.5).astype("int32")
train_dict = get_model_prediction_results(y_pred_train_classes, y_train, 'train')
print(f'Training accuracy: {train_dict["accuracy"]}')

y_pred_val_classes = (model.model.predict(x_val) > 0.5).astype("int32")
val_dict = get_model_prediction_results(y_pred_val_classes, y_val, 'val')
print(f'Validation accuracy: {val_dict["accuracy"]}')

y_pred_test_classes = (model.model.predict(x_test) > 0.5).astype("int32")
test_dict = get_model_prediction_results(y_pred_test_classes, y_test, 'test')
print(f'Test accuracy: {test_dict["accuracy"]}')
