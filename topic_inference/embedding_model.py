import os

LANGUAGEMODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

embedding_model = None
device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))
print("> Using GPU device:", device_id)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer  # noqa

try:
    embedding_model = SentenceTransformer(LANGUAGEMODEL)
    embedding_model.encode('')  # forces more memory to be used
except RuntimeError as e:
    raise RuntimeError(f"Failed starting the model in device {device_id}, "
                       "maybe there is not sufficient memory available. "
                       "Please change the device in 'CUDA_VISIBLE_DEVICES' "
                       "environment variable.")
