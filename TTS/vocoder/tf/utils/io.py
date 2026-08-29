import datetime
import pickle
import tensorflow as tf


def save_checkpoint(model, current_step, epoch, output_path, **kwargs):
    """ Save TF Vocoder model """
    state = {
        'model': model.weights,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
    }
    state.update(kwargs)
    pickle.dump(state, open(output_path, 'wb'))


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in {"dict", "list", "tuple", "set", "int", "float", "str", "bytes", "bool"}:
            return super().find_class(module, name)
        if "numpy" in module:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unsafe global {module}.{name}")


def load_checkpoint(model, checkpoint_path):
    """ Load TF Vocoder model """
    checkpoint = SafeUnpickler(open(checkpoint_path, 'rb')).load()
    chkp_var_dict = {var.name: var.numpy() for var in checkpoint['model']}
    tf_vars = model.weights
    for tf_var in tf_vars:
        layer_name = tf_var.name
        chkp_var_value = chkp_var_dict[layer_name]
        tf.keras.backend.set_value(tf_var, chkp_var_value)
    return model
