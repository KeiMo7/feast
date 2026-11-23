"""Quick latency sanity check: status tracking on vs off.

Writes 200 entities, then times get_online_features with the capability
disabled (legacy) and enabled (status side-car computed).
"""

import sys
import time

sys.path.insert(0, "sdk/python")
sys.path.insert(0, "sdk/python/tests")

from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import OnlineStatusConfig
from feast.utils import _utc_now
from tests.utils.cli_repo_creator import CliRunner, get_example_repo

N_ENTITIES = 200
N_ROUNDS = 20


def main() -> None:
    runner = CliRunner()
    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"), "file"
    ) as store:
        fv = store.get_feature_view(name="driver_locations")
        provider = store._get_provider()
        provider.online_write_batch(
            config=store.config,
            table=fv,
            data=[
                (
                    EntityKeyProto(
                        join_keys=["driver_id"],
                        entity_values=[ValueProto(int64_val=i)],
                    ),
                    {
                        "lat": ValueProto(float_val=float(i)),
                        "lon": ValueProto(string_val=str(i)),
                    },
                    _utc_now(),
                    _utc_now(),
                )
                for i in range(N_ENTITIES)
            ],
            progress=None,
        )
        entity_rows = [{"driver_id": i} for i in range(N_ENTITIES)]
        features = ["driver_locations:lat", "driver_locations:lon"]

        def bench(label):
            # warmup
            store.get_online_features(features=features, entity_rows=entity_rows)
            start = time.perf_counter()
            for _ in range(N_ROUNDS):
                store.get_online_features(features=features, entity_rows=entity_rows)
            elapsed = (time.perf_counter() - start) / N_ROUNDS * 1000
            print(f"{label}: {elapsed:.2f} ms/call ({N_ENTITIES} entities, 2 features)")
            return elapsed

        store.config.online_status_config = OnlineStatusConfig(enabled=False)
        off = bench("status disabled (legacy)")
        store.config.online_status_config = OnlineStatusConfig(enabled=True)
        on = bench("status enabled           ")
        print(f"overhead: {on - off:+.2f} ms/call ({(on - off) / off * 100:+.1f}%)")


if __name__ == "__main__":
    main()
