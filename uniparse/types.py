import numpy as np


class Parser(object):
    _decoder = None
    _backend = None
    backend_name = "dynet"  # defaults to dynet

    def set_decoder(self, decoder):
        self._decoder = decoder

    def set_backend(self, backend):
        self._backend = backend

    def set_loss_function(self, loss_function):
        self.compute_loss = loss_function

    def set_loss_object(self, loss_object):
        self.loss_object = loss_object

    def get_backend_name(self):
        return self.backend_name

    def decode(self, arc_scores, clip=None):
        decode = self._decoder
        arc_scores = self._backend.to_numpy(arc_scores)

        if clip is not None:
            batch_size, n, _ = arc_scores.shape
            result = np.zeros((batch_size, n), dtype=np.int)  # batch, n
            for i in range(batch_size):
                i_len = clip[i]
                tree = decode(arc_scores[i, :i_len, :i_len])
                result[i, :i_len] = tree
        else:
            result = np.array([decode(s) for s in arc_scores])

        return result

    def save_to_file(self, filename: str) -> None:
        raise NotImplementedError("You need to implement the save procedure your self")

    def load_from_file(self, filename: str) -> None:
        raise NotImplementedError("You need to implement the load procedure your self")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You need to implement the forward pas procedure your self")


# callback interface
class Callback(object):
    def on_train_begin(self, info):
        return

    def on_train_end(self, info):
        return

    def on_epoch_begin(self, epoch, info):
        return

    def on_epoch_end(self, epoch, info):
        return

    def on_batch_begin(self, info):
        return

    def on_batch_end(self, info):
        return
