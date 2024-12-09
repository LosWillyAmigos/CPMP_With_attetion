from cpmp_ml.generators import generate_data_v2
from cpmp_ml.generators import generate_data_v3
from cpmp_ml.optimizer import GreedyV1
from cpmp_ml.optimizer import GreedyV2
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import AttentionModel
from attentional_cpmp.utils import connect_to_server
from attentional_cpmp.utils import load_data_mongo
from attentional_cpmp.utils import save_data_mongo
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

db_host = os.environ.get("DB_HOST")
db_user = os.environ.get("DB_USER")
db_password = os.environ.get("DB_PASSWORD")
db_name = os.environ.get("DB_NAME")

MONGO_URI = f'mongodb://{db_user}:{db_password}@{db_host}/?authSource={db_name}'

# Cantidad de stacks
min_S = 5
max_S = 10
H = 5 
sample_size = 200000
lb = 0
optimizer = GreedyV2()

data, labels = generate_data_v2(min_S, max_S, H, sample_size, lb, optimizer, AttentionModel())

cliente = connect_to_server(MONGO_URI)
base_de_datos = cliente['CPMP_With_Attention']

save_data_mongo(base_de_datos.Sx5, data, labels)

cliente.close()

print('Datos creados y guardados en la base de datos')