from attentional_cpmp.utils import load_data_mongo
from attentional_cpmp.utils import connect_to_server
from attentional_cpmp.model import create_model
from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.validations import validate_model
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import os

load_dotenv()

db_host = os.environ.get("DB_HOST")
db_user = os.environ.get("DB_USER")
db_password = os.environ.get("DB_PASSWORD")
db_name = os.environ.get("DB_NAME")

MONGO_URI_SERVER = f'mongodb://{db_user}:{db_password}@{db_host}/?authSource={db_name}'

client_local = connect_to_server(MONGO_URI_SERVER)

base_de_datos = client_local['CPMP_With_Attention']
data_Sx7 = load_data_mongo(base_de_datos["Sx7"])

client_local.close()

tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

model_Sx7 = create_model(
    num_heads= 5,
    num_stacks= 7, 
    H= 7,
    key_dim= 8,
    activation_hide= 'linear',
    list_neurons_feed= [32, 24, 16],
    list_neurons_hide= [32, 24, 16],
)

for _ in range(2):
    for stack in data_Sx7:
        model_Sx7.fit(np.array(data_Sx7[stack]["States"]), np.array(data_Sx7[stack]["Labels"]), batch_size= 32, epochs= 10)

model_Sx7.save("./models/attentional/Sx7/model_Sx7.h5")

validate_model(model_Sx7, GreedyV2(), AttentionModel(), 10, 7, 50, 1000, max_steps= 100)