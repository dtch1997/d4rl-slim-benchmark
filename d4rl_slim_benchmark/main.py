from absl import app
from absl import flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file('config')

def main(_):
    print(_CONFIG.value)

if __name__ == "__main__":
    app.run(main)