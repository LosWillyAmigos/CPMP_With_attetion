from attentional_cpmp.utils import load_data_mongo
from attentional_cpmp.utils import connect_to_server
from attentional_cpmp.model import create_model
from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.utils.adapters import AttentionModel
from cpmp_ml.validations import validate_model
from dotenv import load_dotenv
import tensorflow as tf
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

# Verificar si TensorFlow está utilizando la GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar TensorFlow para usar la primera GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Usando GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No se encontraron GPUs disponibles.")

# Configurar TensorFlow para usar múltiples hilos
"""tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

model_Sx7 = create_model(num_heads= 12, key_dim= 77, H= 7, num_stacks= 6, value_dim= 20, dropout= 0.8026861028496988,
                         activation_feed= 'softplus', activation_hide= 'softplus', n_dropout_hide= 4, n_dropout_feed= 4, 
                         epsilon= 1.436559786744672e-08, list_neurons_feed= [48, 93, 20, 57, 76, 18, 16, 4, 60, 22, 13, 75, 88],
                         list_neurons_hide= [26, 15, 82, 37, 42])

for _ in range(2):
    for stack in data_Sx7:
        model_Sx7.fit(data_Sx7[stack]["States"], data_Sx7[stack]["Labels"], batch_size= 32, epochs= 10)

model_Sx7.save("./models/Sx7/model_Sx7.h5")

validate_model(model_Sx7, GreedyV2(), AttentionModel(), 10, 7, 50, 1000, max_steps= 100)"""