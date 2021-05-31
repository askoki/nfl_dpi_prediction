from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from src.features.helpers.data_load import get_train_val_test_data, get_model_name, get_train_source, get_model_args, \
    get_and_print_distribution
from src.features.helpers.training import plot_training_validation_loss, get_save_best_model, \
    get_early_stopping_definition
from src.models.gru.models import make_gru_model
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
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = 1
weight_for_1 = int(num_negatives / num_positives) if num_positives != 0 else 0

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

weight_for_0, weight_for_1 = compute_class_weight('balanced', classes=[0, 1], y=y_train)
print('New weight for class 0: {:.2f}'.format(weight_for_0))
print('New weight for class 1: {:.2f}'.format(weight_for_1))
class_weight = {0: weight_for_0, 1: weight_for_1}

EPOCHS = DEFAULT_EPOCHS
BATCH_SIZE = DEFAULT_BATCH_SIZE
model = KerasClassifier(
    build_fn=make_gru_model,
    input_dim=(x_train.shape[1], x_train.shape[2]),
    layers=num_layers,
    cells=num_cells,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f'{datetime.now().strftime("%H:%M:%S")} Start training...')
save_best_model = get_save_best_model(
    Models.GRU.value,
    ML_MODEL,
    monitor=monitor
)
early_stopping = get_early_stopping_definition(
    monitor=monitor,
    mode='max',
    patience=DEFAULT_EPOCHS / 4
)
history = model.fit(
    x_train,
    y_train,
    class_weight=class_weight,
    validation_data=(x_val, y_val),
    callbacks=[save_best_model, early_stopping],
)

print(f'{datetime.now().strftime("%H:%M:%S")} End training...')
model.model.summary()

# print("Saving model...")
# model.model.save(os.path.join(MODELS_DIR, f"{ML_MODEL}_gru.h5"))

plot_training_validation_loss(history, ML_MODEL, Models.GRU.value)

y_pred_train_classes = (model.model.predict(x_train) > 0.5).astype("int32")
print(f'Training accuracy: {accuracy_score(y_pred_train_classes, y_train)}')

y_pred_val_classes = (model.model.predict(x_val) > 0.5).astype("int32")
print(f'Validation accuracy: {accuracy_score(y_pred_val_classes, y_val)}')

y_pred_test_classes = (model.model.predict(x_test) > 0.5).astype("int32")
print(f'Test accuracy: {accuracy_score(y_pred_test_classes, y_test)}')
