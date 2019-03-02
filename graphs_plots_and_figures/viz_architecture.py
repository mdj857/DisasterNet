from keras.models import load_model
from keras.utils import plot_model
m = load_model('best_model_damage.h5')
plot_model(m, show_shapes=True, show_layer_names=False, 3to_file='model.png')