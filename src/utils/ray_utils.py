import ray


def get_object_ids():
    return list(ray.objects().keys())


def print_memory_stats():
    gw = ray.worker.get_global_worker()
    gw.dump_object_store_memory_usage()
