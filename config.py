import os

# Define basedir outside of the Config class
basedir = os.path.abspath(os.path.dirname(__file__))

# Path to the .env file
env_path = os.path.join(basedir, '.env')

# Open and read the .env file
with open(env_path) as f:
    for line in f:
        # Ignore lines starting with '#'
        if not line.startswith('#'):
            # Split each line by '=' to separate key and value
            key, value = line.strip().split('=', 1)
            # Set the environment variable
            os.environ[key] = value


# Now environment variables are loaded, you can access them as usual
class Config(object):
    # Form security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', '').replace(
        'postgres://', 'postgresql://') or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Pagination
    POSTS_PER_PAGE = int(os.environ.get('POSTS_PER_PAGE') or 10)

    # Heroku logs
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')


tokenizer_file_path = 'home/tokenizer2.pkl'
label_encoder_file_path = 'home/label_encoder2.pkl'
model_file_path = 'home/best_model2.h5'
