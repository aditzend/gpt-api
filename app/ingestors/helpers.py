import os
import redis
from dotenv import load_dotenv

load_dotenv()
redis_host = os.getenv("REDIS_HOST") or "localhost"
redis_port = os.getenv("REDIS_PORT") or 6379


def connect_to_redis():
    return redis.Redis(host=redis_host, port=redis_port, db=0)


def check_index_existance(index="main") -> bool:
    redis_conn = connect_to_redis()
    index_key = f"indexes:{index}"
    if redis_conn.get(index_key) is None:
        return False
    else:
        return True
