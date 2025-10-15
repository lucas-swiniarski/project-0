import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from datetime import datetime

class TensorBoardLogger:
    """
    A simple wrapper for torch.utils.tensorboard.SummaryWriter.
    """
    def __init__(self, log_dir: str, run_name: str | None = None, overwrite: bool = False):
        """
        Initializes the TensorBoardLogger.

        Args:
            log_dir (str): The base directory where logs will be stored.
            run_name (str, optional): A specific name for this run. If None, a timestamp-based name is created.
                                      Defaults to None.
            overwrite (bool, optional): If True and a directory for `run_name` already exists,
                                        it will be deleted. Defaults to False.
        """
        if run_name is None:
            run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.log_dir = os.path.join(log_dir, run_name)

        if os.path.exists(self.log_dir) and overwrite:
            print(f"Overwriting existing log directory: {self.log_dir}")
            shutil.rmtree(self.log_dir)

        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logger initialized. Logging to: {self.log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Logs a scalar value.

        Args:
            tag (str): The name of the scalar (e.g., 'Loss/train').
            value (float): The value to log.
            step (int): The global step/iteration to associate with the log.
        """
        self.writer.add_scalar(tag, value, step)

    def log_text(self, tag: str, text: str, step: int):
        """
        Logs a text string.

        Args:
            tag (str): The name of the text (e.g., 'Generations/sample').
            text (str): The text to log.
            step (int): The global step/iteration to associate with the log.
        """
        self.writer.add_text(tag, text, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """
        Logs multiple scalar values to the same chart.

        Args:
            main_tag (str): The parent name of the scalars (e.g., 'Loss/eval').
            tag_scalar_dict (dict): A dictionary of tags to scalar values (e.g., {'train': 0.1, 'val': 0.2}).
            step (int): The global step/iteration to associate with the log.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_hparams(self, hparams: dict, metrics: dict):
        """
        Logs hyperparameters and final metrics.
        """
        self.writer.add_hparams(hparams, metrics)

    def close(self):
        """
        Closes the SummaryWriter.
        """
        self.writer.close()
