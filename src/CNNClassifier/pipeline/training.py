from CNNClassifier.components import DataIngestion,PrepareBaseModel,Training
from CNNClassifier.config import ConfigurationManager
from CNNClassifier import logger

logger.info(f"data ingesiton stage started")

config = ConfigurationManager()
data_ingestion_config = config.get_data_ingestion_config()
data_ingestion = DataIngestion(config=data_ingestion_config)
data_ingestion.download_file()
data_ingestion.unzip_and_clean()

logger.info(f"data ingesiton stage completed")


try:
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_base_model_config()
    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()
except Exception as e:
    raise e

try:
    config = ConfigurationManager()
  
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train()
    
except Exception as e:
    raise e