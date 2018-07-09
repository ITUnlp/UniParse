from uniparse.types import Callback, Parser


class TensorboardLoggerCallback(Callback):
    def __init__(self, logger_destination):
        # import is located inside body in to avoid a dependency on tensorboard / tensorflow
        from uniparse.utility.tensorboard_logging import Logger
        self.writer = Logger(logger_destination)
        print("> writing tensorboard to", logger_destination)

    def on_batch_end(self, info):
        global_step = info["global_step"]
        self.writer("train_arc_acc", info["arc_accuracy"], global_step)
        self.writer("train_rel_acc", info["rel_accuracy"], global_step)
        self.writer("train_arc_loss", info["arc_loss"], global_step)
        self.writer("train_rel_loss", info["rel_loss"], global_step)

    def on_epoch_end(self, epoch, info):
        self.writer("dev_uas", info["dev_uas"], epoch)
        self.writer("dev_las", info["dev_las"], epoch)

    def raw_write(self, variable, value):
        self.writer(variable, value, 0)


class ModelSaveCallback(Callback):
    def __init__(self, save_destination, save_after=0):
        self.save_destination = save_destination
        self.save_after = save_after
        print("> saving model to", save_destination, "(after step %d)" % save_after)
        self.best_uas = -1
        self.best_epoch = -1

    def on_epoch_end(self, epoch, info):
        dev_uas = info["dev_uas"]
        model: Parser = info["model"]
        global_step = info["global_step"]

        if dev_uas > self.best_uas:
            if global_step < self.save_after:
                print("skipped saving (below threshold %d < %d)" % (global_step, self.save_after))
                return
            else:
                self.best_uas = dev_uas
                self.best_epoch = epoch
                model.save_to_file(self.save_destination)
                print("saved to", self.save_destination)
